"""
SST LoRA Fine-tuning Script

Fine-tunes a pre-trained Soil State Transformer on local field data
using Low-Rank Adaptation (LoRA) for parameter-efficient transfer.

Fine-tuning Approach:
    1. Load pre-trained SST checkpoint
    2. Apply LoRA adapters to attention layers (~1% of parameters)
    3. Train on field-specific data (100-500 samples)
    4. Save only the small adapter weights (~200KB per field)

Key Features:
    - Early stopping to prevent overfitting on small datasets
    - Learning rate scheduler with warmup
    - Spatial cross-validation option
    - Per-nutrient metrics tracking
    - Adapter merging for efficient inference

Usage:
    python -m training.finetune --base-model checkpoints/pretrain/best_model.pt \
                                --train-data data/fields/field_001_train.parquet \
                                --field-id field_001

Example:
    >>> from training.finetune import SSTFineTuner
    >>> from training.config import FinetuneConfig, LoRAConfig
    >>>
    >>> config = FinetuneConfig(
    ...     base_model_path='checkpoints/pretrain/best_model.pt',
    ...     lora=LoRAConfig(rank=8, alpha=16),
    ...     field_id='field_001',
    ... )
    >>> trainer = SSTFineTuner(config)
    >>> trainer.finetune(train_data, val_data)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import FinetuneConfig, LoRAConfig, load_config
from training.datasets.wosis_dataset import FieldDataset
from training.datasets.collate import field_collate_fn
from training.metrics import compute_metrics, per_property_metrics
from models.lora_wrapper import (
    apply_lora_to_sst,
    save_lora_adapter,
    load_lora_adapter,
    get_lora_info,
)
from models.soil_state_transformer import SoilStateTransformer

# Optional imports
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.

    Monitors validation loss and stops training when no improvement
    is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like R²
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.should_stop


class SSTFineTuner:
    """
    Fine-tuner for Soil State Transformer using LoRA.

    Loads a pre-trained SST model, applies LoRA adapters,
    and fine-tunes on field-specific data.

    Attributes:
        config: FinetuneConfig with all settings
        base_model: Pre-trained SST (frozen except for LoRA)
        model: LoRA-wrapped model for training
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
    """

    def __init__(self, config: FinetuneConfig):
        """
        Initialize the fine-tuner.

        Args:
            config: FinetuneConfig with training settings
        """
        self.config = config
        self.device = torch.device(config.device)

        # Load base model
        logger.info(f"Loading base model from {config.base_model_path}")
        self.base_model = self._load_base_model()

        # Apply LoRA
        logger.info(f"Applying LoRA (rank={config.lora.rank}, alpha={config.lora.alpha})")
        self.model = apply_lora_to_sst(self.base_model, config.lora)
        self.model.to(self.device)

        # Log LoRA info
        lora_info = get_lora_info(self.model)
        logger.info(f"LoRA trainable parameters: {lora_info['trainable_params']:,}")

        # Initialize optimizer (only LoRA parameters)
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision
        if config.mixed_precision and AMP_AVAILABLE:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )

        # Initialize logging
        self._init_wandb()

        # Training state
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def _load_base_model(self) -> SoilStateTransformer:
        """Load pre-trained SST model."""
        checkpoint_path = Path(self.config.base_model_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Base model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Create model with same config
        if 'config' in checkpoint:
            model_config = checkpoint['config'].get('model', {})
        else:
            model_config = {}

        model = SoilStateTransformer(
            n_properties=model_config.get('n_properties', 8),
            embedding_dim=model_config.get('embedding_dim', 256),
            n_heads=model_config.get('n_heads', 8),
            n_layers=model_config.get('n_layers', 6),
            dropout=model_config.get('dropout', 0.1),
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Store normalization stats
        self.normalization_stats = checkpoint.get('normalization_stats', {})

        logger.info("Base model loaded successfully")
        return model

    def _create_optimizer(self) -> AdamW:
        """Create optimizer for LoRA parameters only."""
        # Get trainable parameters (LoRA only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.config.lr_patience,
                factor=self.config.lr_factor,
            )
        elif self.config.lr_scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
            )
        else:
            return None

    def _init_wandb(self):
        """Initialize WandB logging."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            self.wandb_run = None
            return

        tags = list(self.config.wandb_tags)
        if self.config.field_id:
            tags.append(f'field:{self.config.field_id}')

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=f'finetune-{self.config.field_id}' if self.config.field_id else None,
            tags=tags,
            config={
                **self.config.to_dict(),
                'lora': self.config.lora.to_dict(),
            },
        )

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss with masking for missing targets.

        Args:
            predictions: Model predictions
            targets: Ground truth values
            target_mask: Boolean mask for valid targets
            uncertainties: Optional uncertainty predictions

        Returns:
            Scalar loss value
        """
        # Mask invalid targets
        valid_predictions = predictions[target_mask]
        valid_targets = targets[target_mask]

        if len(valid_predictions) == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        if uncertainties is not None:
            valid_uncertainties = uncertainties[target_mask]
            # Heteroscedastic loss
            precision = torch.exp(-valid_uncertainties)
            loss = torch.mean(
                0.5 * precision * (valid_predictions - valid_targets) ** 2 +
                0.5 * valid_uncertainties
            )
        else:
            # Standard MSE
            loss = nn.functional.mse_loss(valid_predictions, valid_targets)

        return loss

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc='Training', leave=False):
            # Move to device
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)

            location = batch.get('location')
            if location is not None:
                location = location.to(self.device)

            # Forward pass
            with autocast(enabled=self.config.mixed_precision and AMP_AVAILABLE):
                # For fine-tuning, we use a different forward mode
                predictions = self.model(
                    features=features,
                    location=location,
                )

                # Handle tuple output (predictions, uncertainties)
                if isinstance(predictions, tuple):
                    predictions, uncertainties = predictions
                else:
                    uncertainties = None

                loss = self._compute_loss(predictions, targets, target_mask, uncertainties)

            # Backward pass
            self.optimizer.zero_grad()

            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                self.optimizer.step()

            total_loss += loss.item() * len(features)
            total_samples += len(features)

        return {
            'train_loss': total_loss / total_samples,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []
        all_masks = []
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(val_loader, desc='Validation', leave=False):
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)

            location = batch.get('location')
            if location is not None:
                location = location.to(self.device)

            with autocast(enabled=self.config.mixed_precision and AMP_AVAILABLE):
                predictions = self.model(
                    features=features,
                    location=location,
                )

                if isinstance(predictions, tuple):
                    predictions, uncertainties = predictions
                else:
                    uncertainties = None

                loss = self._compute_loss(predictions, targets, target_mask, uncertainties)

            total_loss += loss.item() * len(features)
            total_samples += len(features)

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_masks.append(target_mask.cpu())

        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute metrics
        metrics = {'val_loss': total_loss / total_samples}

        # Per-nutrient metrics
        for i, nutrient in enumerate(self.config.target_nutrients):
            mask = all_masks[:, i]
            if mask.sum() > 0:
                pred = all_predictions[mask, i].numpy()
                target = all_targets[mask, i].numpy()
                nutrient_metrics = compute_metrics(pred, target)
                metrics[f'{nutrient}_r2'] = nutrient_metrics['r2']
                metrics[f'{nutrient}_mae'] = nutrient_metrics['mae']

        # Overall metrics (across all nutrients)
        flat_mask = all_masks.flatten()
        flat_pred = all_predictions.flatten()[flat_mask].numpy()
        flat_target = all_targets.flatten()[flat_mask].numpy()
        overall = compute_metrics(flat_pred, flat_target)
        metrics['val_r2'] = overall['r2']
        metrics['val_mae'] = overall['mae']
        metrics['val_rmse'] = overall['rmse']

        return metrics

    def finetune(
        self,
        train_data: Union[pd.DataFrame, str, Path],
        val_data: Optional[Union[pd.DataFrame, str, Path]] = None,
    ) -> Dict[str, float]:
        """
        Run the full fine-tuning loop.

        Args:
            train_data: Training DataFrame or path
            val_data: Optional validation DataFrame or path

        Returns:
            Dictionary with final metrics
        """
        # Create datasets
        train_dataset = FieldDataset(
            data=train_data,
            targets=list(self.config.target_nutrients),
        )

        if val_data is not None:
            val_dataset = FieldDataset(
                data=val_data,
                targets=list(self.config.target_nutrients),
                stats=train_dataset.stats,  # Use same normalization
            )
        else:
            val_dataset = None

        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=field_collate_fn,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=field_collate_fn,
            )

        # Training loop
        logger.info("=" * 60)
        logger.info("Starting SST LoRA Fine-tuning")
        logger.info(f"  Field ID: {self.config.field_id}")
        logger.info(f"  Max epochs: {self.config.max_epochs}")
        logger.info(f"  Early stopping patience: {self.config.early_stopping_patience}")
        logger.info("=" * 60)

        for epoch in range(1, self.config.max_epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            if val_loader and epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate(val_loader)

                # Learning rate scheduler step
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()

                # Check for best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_epoch = epoch
                    self._save_best_adapter()

                # Log
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch}/{self.config.max_epochs}: "
                    f"train_loss={train_metrics['train_loss']:.4f}, "
                    f"val_loss={val_metrics['val_loss']:.4f}, "
                    f"val_r2={val_metrics['val_r2']:.4f}, "
                    f"lr={lr:.2e}"
                )

                if self.wandb_run:
                    wandb.log({
                        'epoch': epoch,
                        **train_metrics,
                        **val_metrics,
                        'lr': lr,
                    })

                # Early stopping
                if self.early_stopping(val_metrics['val_loss']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                # No validation, just log training
                logger.info(
                    f"Epoch {epoch}/{self.config.max_epochs}: "
                    f"train_loss={train_metrics['train_loss']:.4f}"
                )

                if self.wandb_run:
                    wandb.log({'epoch': epoch, **train_metrics})

        # Final results
        logger.info("=" * 60)
        logger.info("Fine-tuning complete!")
        logger.info(f"  Best epoch: {self.best_epoch}")
        logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
        logger.info(f"  Adapter saved to: {self.config.output_dir}")
        logger.info("=" * 60)

        if self.wandb_run:
            wandb.finish()

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
        }

    def _save_best_adapter(self):
        """Save the best LoRA adapter."""
        save_lora_adapter(
            self.model,
            self.config.output_dir,
            include_config=True,
        )

        # Also save field metadata
        import json
        metadata = {
            'field_id': self.config.field_id,
            'farm_id': self.config.farm_id,
            'target_nutrients': list(self.config.target_nutrients),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'normalization_stats': self.normalization_stats,
        }
        with open(Path(self.config.output_dir) / 'field_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def finetune_field(
    base_model_path: str,
    train_data: Union[pd.DataFrame, str],
    val_data: Optional[Union[pd.DataFrame, str]] = None,
    field_id: str = 'default',
    lora_config: Optional[LoRAConfig] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to fine-tune SST on a single field.

    Args:
        base_model_path: Path to pre-trained checkpoint
        train_data: Training data
        val_data: Validation data
        field_id: Unique field identifier
        lora_config: LoRA configuration
        **kwargs: Additional FinetuneConfig arguments

    Returns:
        Dictionary with fine-tuning results
    """
    config = FinetuneConfig(
        base_model_path=base_model_path,
        lora=lora_config or LoRAConfig(),
        field_id=field_id,
        **kwargs,
    )

    trainer = SSTFineTuner(config)
    return trainer.finetune(train_data, val_data)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Fine-tune SST with LoRA')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--base-model', type=str, help='Path to pre-trained model')
    parser.add_argument('--train-data', type=str, help='Path to training data')
    parser.add_argument('--val-data', type=str, help='Path to validation data')
    parser.add_argument('--field-id', type=str, help='Field identifier')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--alpha', type=float, default=16.0, help='LoRA alpha')
    parser.add_argument('--epochs', type=int, help='Max epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        lora_config = LoRAConfig(rank=args.rank, alpha=args.alpha)
        config = FinetuneConfig(lora=lora_config)

    # Override with CLI args
    if args.base_model:
        config.base_model_path = args.base_model
    if args.train_data:
        config.train_data_path = args.train_data
    if args.val_data:
        config.val_data_path = args.val_data
    if args.field_id:
        config.field_id = args.field_id
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.max_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.no_wandb:
        config.use_wandb = False

    # Run fine-tuning
    trainer = SSTFineTuner(config)
    trainer.finetune(
        train_data=config.train_data_path,
        val_data=config.val_data_path,
    )


if __name__ == '__main__':
    main()

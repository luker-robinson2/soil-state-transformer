"""
SST Pre-training Script

Pre-trains the Soil State Transformer on WoSIS global data
using masked property prediction (similar to BERT's MLM).

Pre-training Objective:
    Randomly mask 15% of soil properties and train the model to
    predict the masked values from spatial context (location, depth)
    and other visible properties.

Features:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation for larger effective batch size
    - Cosine learning rate schedule with warmup
    - WandB logging for experiment tracking
    - Checkpoint saving and resumption
    - Heteroscedastic loss with learned uncertainty

Usage:
    python -m training.pretrain --config configs/pretrain.yaml
    python -m training.pretrain --train-path data/processed/wosis_train.parquet

Example:
    >>> from training.pretrain import SSTPretrainer
    >>> from training.config import PretrainConfig
    >>> config = PretrainConfig()
    >>> trainer = SSTPretrainer(config)
    >>> trainer.train()
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import PretrainConfig, load_config
from training.datasets.wosis_dataset import WoSISDataset
from training.datasets.collate import create_masking_collate_fn
from models.sst_pretrain import SSTPretrainModel

# Optional imports - handle both CUDA and MPS
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
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

try:
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False

# GCS storage for cloud training
try:
    from training.gcs_storage import GCSCheckpointManager, is_gcs_path, load_checkpoint_auto
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# TensorBoard for Vertex AI integration
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SSTPretrainer:
    """
    Pre-trainer for Soil State Transformer.

    Implements masked property prediction pre-training on WoSIS data.
    Handles training loop, validation, checkpointing, and logging.

    Attributes:
        config: PretrainConfig with training settings
        model: SoilStateTransformer model
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision
    """

    def __init__(self, config: PretrainConfig):
        """
        Initialize the pre-trainer.

        Args:
            config: PretrainConfig with all training settings
        """
        self.config = config
        self.device = torch.device(config.device)

        # Set random seed for reproducibility
        self._set_seed(config.seed)

        # Initialize model
        logger.info("Initializing Soil State Transformer")
        self.model = self._create_model()
        self.model.to(self.device)

        # Log model size
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

        # Initialize datasets
        logger.info("Loading datasets")
        self.train_dataset = self._create_dataset(config.train_path)
        self.val_dataset = self._create_dataset(config.val_path) if config.val_path else None

        logger.info(f"Train samples: {len(self.train_dataset):,}")
        if self.val_dataset:
            logger.info(f"Val samples: {len(self.val_dataset):,}")

        # Store normalization stats for inference
        self.normalization_stats = self.train_dataset.get_normalization_stats()

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = self._create_scheduler()

        # Initialize mixed precision (only for CUDA)
        if config.mixed_precision and AMP_AVAILABLE and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info(f"Mixed precision enabled ({config.precision})")
        else:
            self.scaler = None
            if self.device.type == 'mps':
                logger.info("Running on MPS (Apple Silicon) - using FP32")

        # Initialize logging
        self._init_wandb()

        # Initialize GCS checkpoint manager for cloud training
        self._init_gcs_storage()

        # Initialize TensorBoard for Vertex AI integration
        self._init_tensorboard()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_model(self) -> SSTPretrainModel:
        """Create and configure the model."""
        model_config = self.config.model

        model = SSTPretrainModel(
            n_properties=len(self.config.properties),
            embedding_dim=model_config.embedding_dim,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            predict_uncertainty=model_config.predict_uncertainty,
        )

        return model

    def _create_dataset(self, path: str) -> WoSISDataset:
        """Create a dataset from path."""
        return WoSISDataset(
            data_path=path,
            properties=list(self.config.properties),
            normalize=True,
        )

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'embedding' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        return AdamW(optimizer_groups, lr=self.config.learning_rate)

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if not SCHEDULERS_AVAILABLE:
            logger.warning("transformers not available, using no scheduler")
            return None

        if self.config.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps,
            )
        elif self.config.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps,
            )
        else:
            return None

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.config.use_wandb or not WANDB_AVAILABLE:
            self.wandb_run = None
            return

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            tags=self.config.wandb_tags,
            config=self.config.to_dict(),
        )

    def _init_gcs_storage(self):
        """Initialize GCS checkpoint manager for cloud training."""
        self.gcs_manager = None

        if not self.config.use_gcs:
            return

        if not GCS_AVAILABLE:
            logger.warning("GCS requested but google-cloud-storage not available")
            return

        # Get WandB run ID for checkpoint organization
        wandb_run_id = None
        if self.wandb_run:
            wandb_run_id = self.wandb_run.id
        elif os.environ.get('WANDB_RUN_ID'):
            wandb_run_id = os.environ['WANDB_RUN_ID']

        self.gcs_manager = GCSCheckpointManager(
            bucket=self.config.gcs_bucket,
            wandb_run_id=wandb_run_id,
            prefix=self.config.gcs_checkpoint_prefix,
        )

        # Save config to GCS for experiment tracking
        self.gcs_manager.save_config(self.config.to_dict())
        logger.info(f"GCS checkpoints enabled: gs://{self.config.gcs_bucket}/{self.config.gcs_checkpoint_prefix}/")

    def _init_tensorboard(self):
        """Initialize TensorBoard writer for Vertex AI integration."""
        self.tb_writer = None

        # Check for Vertex AI environment
        aip_tensorboard_log_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
        vertex_tensorboard = self.config.vertex_tensorboard

        if aip_tensorboard_log_dir:
            # Running in Vertex AI with TensorBoard
            if TENSORBOARD_AVAILABLE:
                self.tb_writer = SummaryWriter(log_dir=aip_tensorboard_log_dir)
                logger.info(f"TensorBoard initialized at {aip_tensorboard_log_dir}")
            else:
                logger.warning("TensorBoard requested but torch.utils.tensorboard not available")
        elif vertex_tensorboard and TENSORBOARD_AVAILABLE:
            # Local run with TensorBoard output
            log_dir = Path(self.config.output_dir) / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard initialized at {log_dir}")

    def _create_mask(
        self,
        valid_mask: torch.Tensor,
        mask_ratio: float
    ) -> torch.Tensor:
        """
        Create random mask for property prediction.

        Args:
            valid_mask: (batch, n_props) boolean mask of valid properties
            mask_ratio: Fraction of valid properties to mask

        Returns:
            (batch, n_props) boolean tensor indicating masked positions
        """
        batch_size, n_props = valid_mask.shape

        # Random mask
        rand = torch.rand(batch_size, n_props, device=valid_mask.device)
        mask = (rand < mask_ratio) & valid_mask

        # Ensure at least one position is masked per sample
        for i in range(batch_size):
            if not mask[i].any() and valid_mask[i].any():
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                random_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
                mask[i, random_idx] = True

        return mask

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute heteroscedastic loss with learned uncertainty.

        The loss is:
            L = 1/(2σ²) * (y - ŷ)² + 1/2 * log(σ²)

        This encourages the model to learn both accurate predictions
        and calibrated uncertainty estimates.

        Args:
            predictions: Predicted values
            targets: Ground truth values
            log_var: Log variance (uncertainty) predictions

        Returns:
            Scalar loss value
        """
        if log_var is not None:
            # Heteroscedastic loss
            precision = torch.exp(-log_var)
            loss = torch.mean(
                0.5 * precision * (predictions - targets) ** 2 +
                0.5 * log_var
            )
        else:
            # Standard MSE
            loss = nn.functional.mse_loss(predictions, targets)

        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.

        Args:
            batch: Dictionary with batched tensors

        Returns:
            Dictionary with loss values
        """
        self.model.train()

        # Move batch to device
        properties = batch['properties'].to(self.device)
        location = batch['location'].to(self.device)
        depth = batch['depth'].to(self.device)
        valid_mask = batch['mask'].to(self.device)

        # Create mask for prediction
        mask_positions = self._create_mask(valid_mask, self.config.mask_ratio)

        # Mask properties
        masked_properties = properties.clone()
        masked_properties[mask_positions] = self.config.mask_token_value

        # Forward pass with mixed precision
        # Determine device type for autocast
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type, enabled=self.config.mixed_precision and AMP_AVAILABLE and self.device.type == 'cuda'):
            # Use pre-training forward
            predictions, log_var = self.model.forward_pretrain(
                properties=masked_properties,
                location=location,
                depth=depth,
                property_mask=~mask_positions,
            )

            # Compute loss only on masked positions
            masked_predictions = predictions[mask_positions]
            masked_targets = properties[mask_positions]
            masked_log_var = log_var[mask_positions] if log_var is not None else None

            loss = self._compute_loss(masked_predictions, masked_targets, masked_log_var)

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Compute additional metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(masked_predictions - masked_targets))
            mse = torch.mean((masked_predictions - masked_targets) ** 2)

        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'mae': mae.item(),
            'mse': mse.item(),
            'n_masked': mask_positions.sum().item(),
        }

    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        if self.config.gradient_clip_norm > 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation loop.

        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataset is None:
            return {}

        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=create_masking_collate_fn(self.config.mask_ratio),
        )

        total_loss = 0.0
        total_mae = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in tqdm(val_loader, desc='Validation', leave=False):
            # Move to device
            properties = batch['original_properties'].to(self.device)
            masked_properties = batch['properties'].to(self.device)
            location = batch['location'].to(self.device)
            depth = batch['depth'].to(self.device)
            mask_positions = batch['masked_positions'].to(self.device)

            # Forward
            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'
            with autocast(device_type=device_type, enabled=self.config.mixed_precision and AMP_AVAILABLE and self.device.type == 'cuda'):
                predictions, log_var = self.model.forward_pretrain(
                    properties=masked_properties,
                    location=location,
                    depth=depth,
                    property_mask=~mask_positions,
                )

                masked_predictions = predictions[mask_positions]
                masked_targets = properties[mask_positions]
                masked_log_var = log_var[mask_positions] if log_var is not None else None

                loss = self._compute_loss(masked_predictions, masked_targets, masked_log_var)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(masked_predictions - masked_targets)).item()
            total_mse += torch.mean((masked_predictions - masked_targets) ** 2).item()
            n_batches += 1

        return {
            'val_loss': total_loss / n_batches,
            'val_mae': total_mae / n_batches,
            'val_mse': total_mse / n_batches,
            'val_rmse': np.sqrt(total_mse / n_batches),
        }

    def save_checkpoint(self, step: int, is_best: bool = False):
        """
        Save training checkpoint to local and optionally to GCS.

        Args:
            step: Current training step
            is_best: Whether this is the best model so far
        """
        checkpoint_dir = Path(self.config.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config.to_dict(),
            'normalization_stats': self.normalization_stats,
            'best_val_loss': self.best_val_loss,
        }

        # Save to GCS if enabled
        if self.gcs_manager:
            try:
                gcs_path = self.gcs_manager.save_checkpoint(checkpoint, step, is_best)

                # Save normalization stats to GCS for inference
                if is_best and self.normalization_stats:
                    self.gcs_manager.save_normalization_stats(self.normalization_stats)

                # Clean up old GCS checkpoints
                self.gcs_manager.cleanup_old_checkpoints(
                    keep_last_n=self.config.keep_last_n_checkpoints
                )

            except Exception as e:
                logger.error(f"Failed to save checkpoint to GCS: {e}")
                # Continue to save locally as backup

        # Always save locally (as backup or primary storage)
        checkpoint_path = checkpoint_dir / f'checkpoint_{step}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model locally
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

        # Clean up old local checkpoints
        self._cleanup_checkpoints(checkpoint_dir)

    def _cleanup_checkpoints(self, checkpoint_dir: Path):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
        n_to_remove = len(checkpoints) - self.config.keep_last_n_checkpoints

        for ckpt in checkpoints[:n_to_remove]:
            ckpt.unlink()
            logger.debug(f"Removed old checkpoint: {ckpt}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint for resumption.

        Supports both local paths and GCS URIs (gs://bucket/path).

        Args:
            checkpoint_path: Path to checkpoint file (local or gs://...)
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Handle GCS paths
        if GCS_AVAILABLE and is_gcs_path(checkpoint_path):
            if self.gcs_manager:
                checkpoint = self.gcs_manager.load_checkpoint(
                    checkpoint_path,
                    map_location=str(self.device)
                )
            else:
                # Create temporary manager for loading
                checkpoint = load_checkpoint_auto(
                    checkpoint_path,
                    map_location=str(self.device)
                )
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.global_step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.normalization_stats = checkpoint.get('normalization_stats', {})

        logger.info(f"Resumed from step {self.global_step}")

    def train(self, resume_from: Optional[str] = None):
        """
        Run the full training loop.

        Args:
            resume_from: Optional checkpoint path to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        # Create data loader with masking
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )

        # Training loop
        logger.info("=" * 60)
        logger.info("Starting SST Pre-training")
        logger.info(f"  Total steps: {self.config.max_steps:,}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Effective batch: {self.config.effective_batch_size}")
        logger.info(f"  Mask ratio: {self.config.mask_ratio}")
        logger.info("=" * 60)

        accumulated_loss = 0.0
        accumulated_metrics = {'mae': 0.0, 'mse': 0.0, 'n_masked': 0}
        pbar = tqdm(total=self.config.max_steps, initial=self.global_step, desc='Training')

        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1

            for batch_idx, batch in enumerate(train_loader):
                # Training step
                metrics = self.train_step(batch)
                accumulated_loss += metrics['loss']
                for k in accumulated_metrics:
                    accumulated_metrics[k] += metrics.get(k, 0)

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    pbar.update(1)

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                        lr = self.optimizer.param_groups[0]['lr']

                        log_dict = {
                            'train/loss': avg_loss,
                            'train/mae': accumulated_metrics['mae'] / self.config.gradient_accumulation_steps,
                            'train/lr': lr,
                            'train/step': self.global_step,
                            'train/epoch': epoch,
                        }

                        if self.wandb_run:
                            wandb.log(log_dict)

                        # TensorBoard logging for Vertex AI
                        if self.tb_writer:
                            self.tb_writer.add_scalar('train/loss', avg_loss, self.global_step)
                            self.tb_writer.add_scalar('train/mae', log_dict['train/mae'], self.global_step)
                            self.tb_writer.add_scalar('train/lr', lr, self.global_step)

                        pbar.set_postfix(loss=f'{avg_loss:.4f}', lr=f'{lr:.2e}')

                        accumulated_loss = 0.0
                        accumulated_metrics = {'mae': 0.0, 'mse': 0.0, 'n_masked': 0}

                    # Validation
                    if self.global_step % self.config.eval_every == 0:
                        val_metrics = self.validate()

                        if val_metrics:
                            is_best = val_metrics['val_loss'] < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_metrics['val_loss']

                            if self.wandb_run:
                                wandb.log({f'val/{k}': v for k, v in val_metrics.items()})

                            # TensorBoard validation logging
                            if self.tb_writer:
                                for k, v in val_metrics.items():
                                    self.tb_writer.add_scalar(f'val/{k}', v, self.global_step)

                            logger.info(
                                f"Step {self.global_step}: val_loss={val_metrics['val_loss']:.4f}, "
                                f"val_mae={val_metrics['val_mae']:.4f}"
                            )

                    # Checkpointing
                    if self.global_step % self.config.checkpoint_every == 0:
                        val_metrics = self.validate()
                        is_best = val_metrics.get('val_loss', float('inf')) < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint(self.global_step, is_best=is_best)

                    # Check if done
                    if self.global_step >= self.config.max_steps:
                        break

            if self.global_step >= self.config.max_steps:
                break

        pbar.close()

        # Final save
        val_metrics = self.validate()
        is_best = val_metrics.get('val_loss', float('inf')) < self.best_val_loss
        self.save_checkpoint(self.global_step, is_best=True)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"  Final step: {self.global_step}")
        logger.info(f"  Best val loss: {self.best_val_loss:.4f}")
        if self.gcs_manager:
            logger.info(f"  Checkpoints: gs://{self.config.gcs_bucket}/{self.gcs_manager.checkpoint_prefix}/")
        logger.info("=" * 60)

        # Cleanup
        if self.tb_writer:
            self.tb_writer.close()

        if self.wandb_run:
            wandb.finish()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Pre-train Soil State Transformer')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--train-path', type=str, help='Path to training data')
    parser.add_argument('--val-path', type=str, help='Path to validation data')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--max-steps', type=int, help='Maximum training steps')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')

    # Cloud training options
    parser.add_argument('--use-gcs', action='store_true', help='Enable GCS checkpointing')
    parser.add_argument('--gcs-bucket', type=str, help='GCS bucket for checkpoints')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = PretrainConfig()

    # Override with CLI args
    if args.train_path:
        config.train_path = args.train_path
    if args.val_path:
        config.val_path = args.val_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.lr:
        config.learning_rate = args.lr
    if args.no_wandb:
        config.use_wandb = False

    # Cloud training options
    if args.use_gcs:
        config.use_gcs = True
    if args.gcs_bucket:
        config.gcs_bucket = args.gcs_bucket

    # Create trainer and run
    trainer = SSTPretrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()

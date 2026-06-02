#!/usr/bin/env python3
"""
Evaluate Pre-trained SST Model

Loads the pre-trained Soil State Transformer and evaluates it on the
WoSIS test set using masked property prediction (same as pre-training objective).

Usage:
    python scripts/evaluate_pretrain.py
    python scripts/evaluate_pretrain.py --checkpoint checkpoints/pretrain/local/best_model.pt
    python scripts/evaluate_pretrain.py --test-data gs://your-research-bucket/data/processed/wosis_test.parquet
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.sst_pretrain import SSTPretrainModel
from training.datasets.wosis_dataset import WoSISDataset
from training.metrics import compute_metrics, per_property_metrics, format_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CHECKPOINT = PROJECT_ROOT / 'checkpoints/pretrain/local/best_model.pt'
DEFAULT_TEST_DATA = 'gs://your-research-bucket/data/processed/wosis_test.parquet'
RESULTS_DIR = PROJECT_ROOT / 'results/pretrain_evaluation'


def load_model(checkpoint_path: Path, device: str = 'cpu') -> Tuple[SSTPretrainModel, Dict]:
    """
    Load pre-trained SST model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded SSTPretrainModel
        config: Training configuration from checkpoint
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    # Create model with saved architecture
    model = SSTPretrainModel(
        n_properties=model_config.get('n_properties', 8),
        embedding_dim=model_config.get('embedding_dim', 256),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        dropout=model_config.get('dropout', 0.1),
        predict_uncertainty=model_config.get('predict_uncertainty', True),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)

    logger.info(f"Model loaded successfully: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, config


def load_test_data(
    data_path: str,
    normalization_stats: Dict,
    properties: Tuple[str, ...],
) -> WoSISDataset:
    """
    Load test dataset with normalization from training.

    Args:
        data_path: Path to test parquet file (local or GCS)
        normalization_stats: Stats from training for consistent normalization
        properties: Property names to load

    Returns:
        WoSISDataset instance
    """
    logger.info(f"Loading test data from {data_path}")

    dataset = WoSISDataset(
        data_path=data_path,
        properties=list(properties),
        normalize=True,
        stats=normalization_stats,
    )

    logger.info(f"Loaded {len(dataset)} test samples")
    return dataset


def evaluate_masked_prediction(
    model: SSTPretrainModel,
    dataset: WoSISDataset,
    device: str,
    mask_ratio: float = 0.15,
    n_eval_passes: int = 5,
    batch_size: int = 256,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on masked property prediction task.

    This mimics the pre-training objective: mask some properties and
    predict them from context.

    Args:
        model: Pre-trained SST model
        dataset: WoSIS test dataset
        device: Device to run on
        mask_ratio: Fraction of properties to mask
        n_eval_passes: Number of evaluation passes with different masks
        batch_size: Batch size for evaluation

    Returns:
        metrics: Dictionary of evaluation metrics
        all_predictions: Stacked predictions across passes
        all_targets: Stacked targets
        all_log_var: Stacked log variance predictions
    """
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_properties = model.n_properties
    all_predictions = []
    all_targets = []
    all_log_vars = []
    all_masks = []  # Which properties were masked

    with torch.no_grad():
        for pass_idx in range(n_eval_passes):
            logger.info(f"Evaluation pass {pass_idx + 1}/{n_eval_passes}")

            pass_preds = []
            pass_targets = []
            pass_log_vars = []
            pass_masks = []

            for batch in tqdm(dataloader, desc=f"Pass {pass_idx + 1}"):
                properties = batch['properties'].to(device)
                location = batch['location'].to(device)
                depth = batch['depth'].to(device)
                valid_mask = batch['mask'].to(device)  # True where property is valid

                batch_size_actual = properties.size(0)

                # Create random mask (True = visible, False = masked for prediction)
                random_mask = torch.rand(batch_size_actual, n_properties, device=device) > mask_ratio

                # Only mask valid properties
                prediction_mask = valid_mask & ~random_mask  # True = should predict
                visible_mask = valid_mask & random_mask  # True = visible as input

                # Mask properties for input
                masked_properties = properties.clone()
                masked_properties[~visible_mask] = 0.0  # Set masked to 0

                # Forward pass
                predictions, log_var = model.forward_pretrain(
                    masked_properties,
                    location,
                    depth,
                    visible_mask,  # Only show visible properties
                )

                # Collect predictions only for masked positions
                pass_preds.append(predictions.cpu().numpy())
                pass_targets.append(properties.cpu().numpy())
                pass_log_vars.append(log_var.cpu().numpy() if log_var is not None else None)
                pass_masks.append(prediction_mask.cpu().numpy())

            all_predictions.append(np.concatenate(pass_preds, axis=0))
            all_targets.append(np.concatenate(pass_targets, axis=0))
            if pass_log_vars[0] is not None:
                all_log_vars.append(np.concatenate(pass_log_vars, axis=0))
            all_masks.append(np.concatenate(pass_masks, axis=0))

    # Stack across passes
    all_predictions = np.stack(all_predictions, axis=0)  # (n_passes, n_samples, n_props)
    all_targets = np.stack(all_targets, axis=0)
    all_masks = np.stack(all_masks, axis=0)
    if all_log_vars:
        all_log_vars = np.stack(all_log_vars, axis=0)

    # Compute metrics on masked positions only
    metrics = {}
    property_names = list(dataset.properties)

    for i, prop in enumerate(property_names):
        # Get predictions only where this property was masked
        mask = all_masks[..., i]  # (n_passes, n_samples)
        preds = all_predictions[..., i][mask]
        targets = all_targets[..., i][mask]

        if len(preds) > 0:
            prop_metrics = compute_metrics(preds, targets)
            metrics[prop] = prop_metrics
            logger.info(f"{prop}: R²={prop_metrics['r2']:.4f}, MAE={prop_metrics['mae']:.4f}")

    # Overall metrics (all properties combined)
    overall_preds = all_predictions[all_masks]
    overall_targets = all_targets[all_masks]
    overall_metrics = compute_metrics(overall_preds, overall_targets)
    metrics['overall'] = overall_metrics

    logger.info(f"\nOverall: R²={overall_metrics['r2']:.4f}, MAE={overall_metrics['mae']:.4f}")

    log_vars_result = np.stack(all_log_vars, axis=0) if len(all_log_vars) > 0 else None
    return metrics, all_predictions, all_targets, log_vars_result


def evaluate_reconstruction(
    model: SSTPretrainModel,
    dataset: WoSISDataset,
    device: str,
    batch_size: int = 256,
) -> Dict:
    """
    Evaluate model's ability to reconstruct all properties (no masking).

    This tests if the model learned useful representations.

    Args:
        model: Pre-trained SST model
        dataset: WoSIS test dataset
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        metrics: Dictionary of reconstruction metrics per property
    """
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    all_predictions = []
    all_targets = []
    all_log_vars = []
    all_valid = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Reconstruction"):
            properties = batch['properties'].to(device)
            location = batch['location'].to(device)
            depth = batch['depth'].to(device)
            valid_mask = batch['mask'].to(device)

            # All properties visible
            full_mask = torch.ones_like(valid_mask)

            predictions, log_var = model.forward_pretrain(
                properties,
                location,
                depth,
                full_mask,
            )

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(properties.cpu().numpy())
            if log_var is not None:
                all_log_vars.append(log_var.cpu().numpy())
            all_valid.append(valid_mask.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    valid = np.concatenate(all_valid, axis=0)
    if all_log_vars:
        log_vars = np.concatenate(all_log_vars, axis=0)
    else:
        log_vars = None

    # Per-property metrics
    property_names = list(dataset.properties)
    metrics = per_property_metrics(
        predictions, targets, property_names,
        log_vars if log_vars is not None else None,
    )

    # Filter by valid values
    for i, prop in enumerate(property_names):
        mask = valid[:, i]
        preds = predictions[mask, i]
        targs = targets[mask, i]
        log_v = log_vars[mask, i] if log_vars is not None else None
        metrics[prop] = compute_metrics(preds, targs, log_v)

    return metrics


def compute_baseline_metrics(
    dataset: WoSISDataset,
) -> Dict:
    """
    Compute baseline metrics (mean prediction).

    The mean predictor just predicts the training mean for each property.
    This serves as a sanity check - our model should beat this.

    Args:
        dataset: WoSIS test dataset

    Returns:
        metrics: Baseline metrics per property
    """
    property_names = list(dataset.properties)
    metrics = {}

    for i, prop in enumerate(property_names):
        # Get valid values
        mask = dataset._masks[:, i]
        targets = dataset._properties[mask, i]

        # Mean prediction (which is 0 for normalized data)
        predictions = np.zeros_like(targets)

        metrics[prop] = compute_metrics(predictions, targets)

    return metrics


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: list,
    output_dir: Path,
    dataset: WoSISDataset,
):
    """
    Generate scatter plots of predictions vs targets.

    Args:
        predictions: (n_samples, n_properties) predictions
        targets: (n_samples, n_properties) targets
        property_names: Names of properties
        output_dir: Directory to save plots
        dataset: Dataset for denormalization
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_props = len(property_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (prop, ax) in enumerate(zip(property_names, axes)):
        if i >= n_props:
            ax.set_visible(False)
            continue

        pred = predictions[:, i]
        targ = targets[:, i]

        # Remove NaN
        mask = ~(np.isnan(pred) | np.isnan(targ))
        pred = pred[mask]
        targ = targ[mask]

        if len(pred) == 0:
            ax.set_visible(False)
            continue

        # Denormalize for interpretable units
        if dataset.stats and prop in dataset.stats:
            pred = pred * dataset.stats[prop]['std'] + dataset.stats[prop]['mean']
            targ = targ * dataset.stats[prop]['std'] + dataset.stats[prop]['mean']

        # Scatter plot
        ax.scatter(targ, pred, alpha=0.3, s=5)

        # Perfect prediction line
        lims = [
            min(targ.min(), pred.min()),
            max(targ.max(), pred.max())
        ]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')

        # Metrics
        r2 = compute_metrics(pred, targ)['r2']
        mae = compute_metrics(pred, targ)['mae']

        ax.set_xlabel(f'Actual {prop}')
        ax.set_ylabel(f'Predicted {prop}')
        ax.set_title(f'{prop}\nR²={r2:.3f}, MAE={mae:.3f}')
        ax.legend(loc='upper left', fontsize=8)

    # Hide extra axes
    for j in range(n_props, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_scatter.png', dpi=150)
    plt.close()

    logger.info(f"Saved scatter plots to {output_dir / 'prediction_scatter.png'}")


def plot_error_distributions(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: list,
    output_dir: Path,
):
    """Generate error distribution histograms."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_props = len(property_names)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (prop, ax) in enumerate(zip(property_names, axes)):
        if i >= n_props:
            ax.set_visible(False)
            continue

        pred = predictions[:, i]
        targ = targets[:, i]

        mask = ~(np.isnan(pred) | np.isnan(targ))
        errors = pred[mask] - targ[mask]

        if len(errors) == 0:
            ax.set_visible(False)
            continue

        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=1)
        ax.axvline(errors.mean(), color='g', linestyle='-', linewidth=1, label=f'Mean={errors.mean():.3f}')

        ax.set_xlabel('Prediction Error (normalized)')
        ax.set_ylabel('Count')
        ax.set_title(f'{prop}\nBias={errors.mean():.3f}, Std={errors.std():.3f}')
        ax.legend(fontsize=8)

    for j in range(n_props, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150)
    plt.close()

    logger.info(f"Saved error distributions to {output_dir / 'error_distributions.png'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate pre-trained SST model')
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT),
                        help='Path to model checkpoint')
    parser.add_argument('--test-data', type=str, default=DEFAULT_TEST_DATA,
                        help='Path to test data (local or GCS)')
    parser.add_argument('--output-dir', type=str, default=str(RESULTS_DIR),
                        help='Directory for results')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to run on (cuda, mps, cpu)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--n-eval-passes', type=int, default=5,
                        help='Number of masked evaluation passes')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    elif args.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load model
    model, config = load_model(Path(args.checkpoint), device)

    # Get normalization stats from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    norm_stats = checkpoint.get('normalization_stats', {})
    properties = tuple(config.get('properties', [
        'clay', 'sand', 'silt', 'phaq', 'orgc', 'nitkjd', 'cecph7', 'bdfi33'
    ]))

    # Load test data
    dataset = load_test_data(args.test_data, norm_stats, properties)

    # Evaluate masked prediction
    logger.info("\n" + "="*60)
    logger.info("Evaluating Masked Property Prediction")
    logger.info("="*60)

    masked_metrics, predictions, targets, log_vars = evaluate_masked_prediction(
        model, dataset, device,
        mask_ratio=0.15,
        n_eval_passes=args.n_eval_passes,
        batch_size=args.batch_size,
    )

    # Evaluate reconstruction
    logger.info("\n" + "="*60)
    logger.info("Evaluating Reconstruction (All Properties Visible)")
    logger.info("="*60)

    recon_metrics = evaluate_reconstruction(
        model, dataset, device,
        batch_size=args.batch_size,
    )

    for prop, metrics in recon_metrics.items():
        logger.info(f"{prop}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")

    # Baseline comparison
    logger.info("\n" + "="*60)
    logger.info("Computing Baseline (Mean Prediction)")
    logger.info("="*60)

    baseline_metrics = compute_baseline_metrics(dataset)
    for prop, metrics in baseline_metrics.items():
        logger.info(f"{prop}: R²={metrics['r2']:.4f} (mean predictor)")

    # Generate plots
    logger.info("\n" + "="*60)
    logger.info("Generating Visualizations")
    logger.info("="*60)

    # Use first pass predictions for plotting
    plot_predictions(
        predictions[0], targets[0],
        list(properties), output_dir, dataset
    )

    plot_error_distributions(
        predictions[0], targets[0],
        list(properties), output_dir
    )

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'test_data': args.test_data,
        'n_samples': len(dataset),
        'n_eval_passes': args.n_eval_passes,
        'mask_ratio': 0.15,
        'training_config': {
            'step': checkpoint.get('step'),
            'best_val_loss': checkpoint.get('best_val_loss'),
        },
        'masked_prediction_metrics': {k: v for k, v in masked_metrics.items()},
        'reconstruction_metrics': recon_metrics,
        'baseline_metrics': baseline_metrics,
    }

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results = convert_numpy(results)

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Model: {args.checkpoint}")
    logger.info(f"Test samples: {len(dataset)}")
    logger.info(f"Training step: {checkpoint.get('step')}")
    logger.info(f"Best val loss: {checkpoint.get('best_val_loss'):.4f}")
    logger.info("")
    logger.info("Per-Property Masked Prediction R²:")
    for prop in properties:
        if prop in masked_metrics:
            r2 = masked_metrics[prop]['r2']
            mae = masked_metrics[prop]['mae']
            logger.info(f"  {prop:10s}: R²={r2:.4f}, MAE={mae:.4f}")
    logger.info("")
    if 'overall' in masked_metrics:
        logger.info(f"Overall: R²={masked_metrics['overall']['r2']:.4f}, MAE={masked_metrics['overall']['mae']:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

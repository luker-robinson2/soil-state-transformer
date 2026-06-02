"""
Evaluation Metrics for Soil Property Prediction

Provides comprehensive metrics for evaluating soil property predictions:
- Standard regression metrics (R², MAE, RMSE, MSE)
- Uncertainty calibration metrics
- Per-property metrics
- Spatial generalization metrics
- Comparison with baseline models

Usage:
    >>> from training.metrics import compute_metrics, per_property_metrics
    >>> metrics = compute_metrics(predictions, targets)
    >>> print(f"R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Optional sklearn imports
try:
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive regression metrics.

    Args:
        predictions: Model predictions, shape (n_samples,) or (n_samples, n_outputs)
        targets: Ground truth values, same shape as predictions
        uncertainties: Optional uncertainty estimates (log variance or std dev)

    Returns:
        Dictionary with metrics:
            - r2: R² score (coefficient of determination)
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - mse: Mean squared error
            - mape: Mean absolute percentage error (if targets > 0)
            - bias: Mean prediction bias
            - std_residual: Standard deviation of residuals
            - nll: Negative log likelihood (if uncertainties provided)
            - calibration: Calibration error (if uncertainties provided)
    """
    # Flatten if needed
    predictions = np.asarray(predictions).flatten()
    targets = np.asarray(targets).flatten()

    # Remove NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    if len(predictions) == 0:
        return {k: float('nan') for k in ['r2', 'mae', 'rmse', 'mse', 'bias', 'std_residual']}

    # Basic metrics
    residuals = predictions - targets

    metrics = {
        'r2': _compute_r2(predictions, targets),
        'mae': float(np.mean(np.abs(residuals))),
        'mse': float(np.mean(residuals ** 2)),
        'rmse': float(np.sqrt(np.mean(residuals ** 2))),
        'bias': float(np.mean(residuals)),
        'std_residual': float(np.std(residuals)),
        'n_samples': len(predictions),
    }

    # MAPE (only if targets are positive)
    if np.all(targets > 0):
        metrics['mape'] = float(np.mean(np.abs(residuals / targets)) * 100)

    # Uncertainty metrics
    if uncertainties is not None:
        uncertainties = np.asarray(uncertainties).flatten()[valid_mask]
        metrics.update(_compute_uncertainty_metrics(predictions, targets, uncertainties))

    return metrics


def _compute_r2(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute R² score."""
    if SKLEARN_AVAILABLE:
        return float(r2_score(targets, predictions))
    else:
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)


def _compute_uncertainty_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    log_var: np.ndarray,
) -> Dict[str, float]:
    """
    Compute uncertainty-related metrics.

    Args:
        predictions: Point predictions
        targets: Ground truth
        log_var: Log variance predictions

    Returns:
        Dictionary with uncertainty metrics
    """
    # Convert log variance to standard deviation
    std = np.sqrt(np.exp(log_var))

    # Negative log likelihood (Gaussian)
    nll = 0.5 * (log_var + ((targets - predictions) ** 2) * np.exp(-log_var))
    nll = float(np.mean(nll))

    # Calibration error
    # For well-calibrated predictions, ~68% of targets should fall within ±1 std
    within_1std = np.mean(np.abs(targets - predictions) <= std)
    within_2std = np.mean(np.abs(targets - predictions) <= 2 * std)

    # Expected: 68.27% within 1std, 95.45% within 2std
    calibration_1std = abs(within_1std - 0.6827)
    calibration_2std = abs(within_2std - 0.9545)

    # Sharpness (average prediction uncertainty)
    sharpness = float(np.mean(std))

    return {
        'nll': nll,
        'calibration_1std': float(calibration_1std),
        'calibration_2std': float(calibration_2std),
        'within_1std': float(within_1std),
        'within_2std': float(within_2std),
        'sharpness': sharpness,
    }


def per_property_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    property_names: List[str],
    uncertainties: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each soil property separately.

    Args:
        predictions: Shape (n_samples, n_properties)
        targets: Shape (n_samples, n_properties)
        property_names: Names of properties
        uncertainties: Optional uncertainties, same shape

    Returns:
        Dictionary mapping property names to their metrics
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    results = {}
    for i, prop in enumerate(property_names):
        prop_uncertainties = uncertainties[:, i] if uncertainties is not None else None
        results[prop] = compute_metrics(
            predictions[:, i],
            targets[:, i],
            prop_uncertainties,
        )

    return results


def compute_spatial_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    spatial_groups: np.ndarray,
) -> Dict[str, float]:
    """
    Compute metrics that assess spatial generalization.

    Useful for understanding how well the model generalizes to
    new spatial locations.

    Args:
        predictions: Model predictions
        targets: Ground truth
        spatial_groups: Array of spatial group IDs

    Returns:
        Dictionary with spatial metrics
    """
    unique_groups = np.unique(spatial_groups)
    group_r2s = []
    group_maes = []

    for group in unique_groups:
        mask = spatial_groups == group
        if mask.sum() < 5:  # Skip very small groups
            continue

        group_metrics = compute_metrics(predictions[mask], targets[mask])
        group_r2s.append(group_metrics['r2'])
        group_maes.append(group_metrics['mae'])

    return {
        'mean_group_r2': float(np.mean(group_r2s)),
        'std_group_r2': float(np.std(group_r2s)),
        'min_group_r2': float(np.min(group_r2s)),
        'max_group_r2': float(np.max(group_r2s)),
        'mean_group_mae': float(np.mean(group_maes)),
        'n_groups': len(group_r2s),
    }


def compare_to_baseline(
    model_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compare model predictions to a baseline.

    Args:
        model_predictions: Predictions from the model
        baseline_predictions: Predictions from baseline (e.g., mean, XGBoost)
        targets: Ground truth values

    Returns:
        Dictionary with comparison metrics
    """
    model_metrics = compute_metrics(model_predictions, targets)
    baseline_metrics = compute_metrics(baseline_predictions, targets)

    # Improvement ratios
    r2_improvement = (model_metrics['r2'] - baseline_metrics['r2'])
    mae_improvement = (baseline_metrics['mae'] - model_metrics['mae']) / baseline_metrics['mae'] * 100
    rmse_improvement = (baseline_metrics['rmse'] - model_metrics['rmse']) / baseline_metrics['rmse'] * 100

    return {
        'model_r2': model_metrics['r2'],
        'baseline_r2': baseline_metrics['r2'],
        'r2_improvement': r2_improvement,
        'model_mae': model_metrics['mae'],
        'baseline_mae': baseline_metrics['mae'],
        'mae_improvement_pct': mae_improvement,
        'model_rmse': model_metrics['rmse'],
        'baseline_rmse': baseline_metrics['rmse'],
        'rmse_improvement_pct': rmse_improvement,
    }


def compute_cross_validation_metrics(
    fold_results: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate metrics across cross-validation folds.

    Args:
        fold_results: List of per-fold metric dictionaries

    Returns:
        Dictionary with aggregated statistics
    """
    if not fold_results:
        return {}

    # Get all metric names
    metric_names = fold_results[0].keys()

    aggregated = {}
    for name in metric_names:
        if name == 'n_samples':
            continue

        values = [f[name] for f in fold_results if not np.isnan(f.get(name, np.nan))]
        if values:
            aggregated[f'{name}_mean'] = float(np.mean(values))
            aggregated[f'{name}_std'] = float(np.std(values))
            aggregated[f'{name}_min'] = float(np.min(values))
            aggregated[f'{name}_max'] = float(np.max(values))

    aggregated['n_folds'] = len(fold_results)
    return aggregated


def format_metrics_table(
    metrics: Dict[str, float],
    property_names: Optional[List[str]] = None,
) -> str:
    """
    Format metrics as a readable table.

    Args:
        metrics: Dictionary of metrics
        property_names: If provided, format per-property metrics

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Evaluation Metrics")
    lines.append("=" * 50)

    # Overall metrics
    overall_keys = ['r2', 'mae', 'rmse', 'mse', 'bias', 'n_samples']
    lines.append("\nOverall:")
    for key in overall_keys:
        if key in metrics:
            value = metrics[key]
            if key == 'n_samples':
                lines.append(f"  {key}: {value:,}")
            else:
                lines.append(f"  {key}: {value:.4f}")

    # Uncertainty metrics
    uncertainty_keys = ['nll', 'sharpness', 'within_1std', 'within_2std']
    uncertainty_present = any(k in metrics for k in uncertainty_keys)
    if uncertainty_present:
        lines.append("\nUncertainty:")
        for key in uncertainty_keys:
            if key in metrics:
                lines.append(f"  {key}: {metrics[key]:.4f}")

    lines.append("=" * 50)
    return "\n".join(lines)


class MetricsTracker:
    """
    Track metrics over training epochs.

    Useful for monitoring training progress and detecting overfitting.
    """

    def __init__(self):
        self.history = {}

    def update(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Add metrics for a step."""
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append({
                'step': step,
                'value': value,
            })

    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[int, float]:
        """Get the step and value of best metric."""
        if metric_name not in self.history:
            return None, None

        values = [h['value'] for h in self.history[metric_name]]
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        return self.history[metric_name][best_idx]['step'], values[best_idx]

    def get_trend(self, metric_name: str, window: int = 5) -> float:
        """Get trend of metric (positive = improving for min metrics)."""
        if metric_name not in self.history or len(self.history[metric_name]) < window:
            return 0.0

        values = [h['value'] for h in self.history[metric_name][-window:]]
        # Linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)

    def summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.history.items():
            vals = [h['value'] for h in values]
            summary[name] = {
                'current': vals[-1] if vals else None,
                'best': min(vals) if 'loss' in name else max(vals),
                'mean': np.mean(vals),
                'std': np.std(vals),
            }
        return summary

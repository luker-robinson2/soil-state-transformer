"""
Training Utilities

Provides utility functions for training:
- Checkpoint loading and saving
- Logging setup
- Device detection
- Reproducibility helpers
- Learning rate scheduling utilities
- Gradient checking

Usage:
    >>> from training.utils import setup_device, set_seed, load_checkpoint
    >>> device = setup_device()
    >>> set_seed(42)
    >>> model, optimizer, step = load_checkpoint('checkpoint.pt', model, optimizer)
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


# =============================================================================
# Device Setup
# =============================================================================

def setup_device(device: Optional[str] = None) -> torch.device:
    """
    Set up compute device with appropriate backend.

    Automatically detects available hardware and sets up accordingly.

    Args:
        device: Optional device string ('cuda', 'mps', 'cpu')

    Returns:
        torch.device for computation
    """
    if device:
        return torch.device(device)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Metal (MPS)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")

    return device


def get_device_info() -> Dict[str, Any]:
    """Get detailed device information."""
    info = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    return info


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # For reproducibility with cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.debug(f"Set random seed to {seed}")


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
    extra: Optional[Dict] = None,
):
    """
    Save training checkpoint.

    Args:
        path: Save path
        model: Model to save
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        step: Current training step
        epoch: Current epoch
        metrics: Optional metrics dictionary
        config: Optional configuration dictionary
        extra: Optional extra data to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'step': step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if metrics is not None:
        checkpoint['metrics'] = metrics

    if config is not None:
        checkpoint['config'] = config

    if extra is not None:
        checkpoint.update(extra)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[nn.Module, Optional[Optimizer], int]:
    """
    Load training checkpoint.

    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load tensors to
        strict: Whether to require exact state dict match

    Returns:
        Tuple of (model, optimizer, step)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device or 'cpu')

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    step = checkpoint.get('step', 0)
    logger.info(f"Loaded checkpoint from step {step}")

    return model, optimizer, step


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search

    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(p):
        try:
            return int(p.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0

    return max(checkpoints, key=get_step)


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
):
    """
    Configure logging for training.

    Args:
        log_file: Optional file to write logs to
        level: Logging level
        format_str: Custom format string
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True,
    )


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': trainable / total * 100 if total > 0 else 0,
    }


def freeze_model(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze all model parameters.

    Args:
        model: Model to freeze/unfreeze
        freeze: If True, freeze parameters; if False, unfreeze
    """
    for param in model.parameters():
        param.requires_grad = not freeze


def freeze_layers(model: nn.Module, layer_names: List[str], freeze: bool = True):
    """
    Freeze specific layers by name.

    Args:
        model: Model
        layer_names: List of layer name patterns to freeze
        freeze: If True, freeze; if False, unfreeze
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in layer_names):
            param.requires_grad = not freeze


def get_grad_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm.

    Args:
        model: Model with computed gradients

    Returns:
        Total L2 norm of gradients
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# =============================================================================
# Learning Rate Utilities
# =============================================================================

def get_lr(optimizer: Optimizer) -> float:
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def set_lr(optimizer: Optimizer, lr: float):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """
    Compute learning rate with linear warmup.

    Args:
        step: Current step
        warmup_steps: Number of warmup steps
        base_lr: Target learning rate after warmup

    Returns:
        Learning rate for current step
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


class WarmupScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
        max_steps: int = 100000,
        schedule: str = 'cosine',
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.schedule = schedule
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        lr = self._get_lr()
        set_lr(self.optimizer, lr)
        return lr

    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps

        # After warmup
        progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = min(1.0, progress)

        if self.schedule == 'cosine':
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        elif self.schedule == 'linear':
            return self.base_lr - (self.base_lr - self.min_lr) * progress
        else:
            return self.base_lr


# =============================================================================
# Data Utilities
# =============================================================================

def split_data(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/val/test sets.

    Args:
        data: Data to split (indices or array)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train, val, test) indices/arrays
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(data)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if isinstance(data, np.ndarray):
        return data[train_idx], data[val_idx], data[test_idx]
    return train_idx, val_idx, test_idx


# =============================================================================
# Experiment Tracking
# =============================================================================

class ExperimentLogger:
    """
    Simple experiment logger that writes to JSON files.

    Alternative to WandB for offline logging.
    """

    def __init__(self, log_dir: Union[str, Path], experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f'{experiment_name}.jsonl'
        self.config_file = self.log_dir / f'{experiment_name}_config.json'

        self.step = 0

    def log_config(self, config: Dict):
        """Log configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics for a step."""
        if step is not None:
            self.step = step

        entry = {
            'step': self.step,
            'timestamp': datetime.now().isoformat(),
            **metrics,
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        self.step += 1

    def load_history(self) -> List[Dict]:
        """Load logged history."""
        if not self.log_file.exists():
            return []

        history = []
        with open(self.log_file) as f:
            for line in f:
                history.append(json.loads(line))
        return history

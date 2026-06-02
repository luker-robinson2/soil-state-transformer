"""
Training scripts for Foundation Model Research.

This module provides:
- PretrainConfig/LoRAConfig/FinetuneConfig: Training configurations
- SSTPretrainer: Pre-training on global WoSIS data
- SSTFineTuner: LoRA fine-tuning on local field data
- Metrics and utilities for evaluation and monitoring

Usage:
    # Pre-training
    >>> from training import SSTPretrainer, PretrainConfig
    >>> config = PretrainConfig(train_path='data/wosis_train.parquet')
    >>> trainer = SSTPretrainer(config)
    >>> trainer.train()

    # Fine-tuning
    >>> from training import SSTFineTuner, FinetuneConfig, LoRAConfig
    >>> config = FinetuneConfig(
    ...     base_model_path='checkpoints/best_model.pt',
    ...     lora=LoRAConfig(rank=8),
    ...     field_id='field_001',
    ... )
    >>> trainer = SSTFineTuner(config)
    >>> trainer.finetune(train_data, val_data)
"""

# Configuration classes
from training.config import (
    ModelConfig,
    PretrainConfig,
    LoRAConfig,
    FinetuneConfig,
    InferenceConfig,
    load_config,
    save_config,
)

# Training classes
from training.pretrain import SSTPretrainer
from training.finetune import SSTFineTuner, finetune_field

# Metrics
from training.metrics import (
    compute_metrics,
    per_property_metrics,
    compute_spatial_metrics,
    compare_to_baseline,
    compute_cross_validation_metrics,
    format_metrics_table,
    MetricsTracker,
)

# Utilities
from training.utils import (
    setup_device,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    setup_logging,
    count_parameters,
    freeze_model,
    freeze_layers,
    get_lr,
    set_lr,
    WarmupScheduler,
    ExperimentLogger,
)

# Datasets
from training.datasets import (
    WoSISDataset,
    FieldDataset,
    wosis_collate_fn,
    field_collate_fn,
)

__all__ = [
    # Config
    'ModelConfig',
    'PretrainConfig',
    'LoRAConfig',
    'FinetuneConfig',
    'InferenceConfig',
    'load_config',
    'save_config',
    # Trainers
    'SSTPretrainer',
    'SSTFineTuner',
    'finetune_field',
    # Metrics
    'compute_metrics',
    'per_property_metrics',
    'compute_spatial_metrics',
    'compare_to_baseline',
    'compute_cross_validation_metrics',
    'format_metrics_table',
    'MetricsTracker',
    # Utilities
    'setup_device',
    'set_seed',
    'save_checkpoint',
    'load_checkpoint',
    'get_latest_checkpoint',
    'setup_logging',
    'count_parameters',
    'freeze_model',
    'freeze_layers',
    'get_lr',
    'set_lr',
    'WarmupScheduler',
    'ExperimentLogger',
    # Datasets
    'WoSISDataset',
    'FieldDataset',
    'wosis_collate_fn',
    'field_collate_fn',
]

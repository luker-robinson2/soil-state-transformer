"""
Training Datasets for Foundation Model Research.

This module provides PyTorch Dataset classes for soil data:
- WoSISDataset: Global WoSIS soil profile data for pre-training
- FieldDataset: Local field-specific data for fine-tuning

Custom collation functions handle variable-length sequences and
missing property values.
"""

from training.datasets.wosis_dataset import WoSISDataset, FieldDataset
from training.datasets.collate import (
    wosis_collate_fn,
    field_collate_fn,
    create_masking_collate_fn,
    MaskingCollator,
)

__all__ = [
    'WoSISDataset',
    'FieldDataset',
    'wosis_collate_fn',
    'field_collate_fn',
    'create_masking_collate_fn',
    'MaskingCollator',
]

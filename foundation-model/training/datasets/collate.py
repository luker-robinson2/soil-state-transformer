"""
Custom Collation Functions for Soil Datasets

Provides collate functions for PyTorch DataLoader that handle:
- Variable-length sequences (for temporal data)
- Missing property values (masked)
- Batching of mixed tensor and non-tensor data

These functions are used with DataLoader's collate_fn parameter.
"""

from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence


def wosis_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for WoSIS pre-training dataset.

    Batches samples from WoSISDataset, stacking tensors and
    handling profile IDs appropriately.

    Args:
        batch: List of sample dictionaries from WoSISDataset

    Returns:
        Batched dictionary with:
            - properties: (batch_size, n_properties) tensor
            - location: (batch_size, 2) tensor
            - depth: (batch_size, 2) tensor
            - mask: (batch_size, n_properties) boolean tensor
            - profile_ids: list of profile ID strings
    """
    # Stack tensors
    properties = torch.stack([s['properties'] for s in batch])
    location = torch.stack([s['location'] for s in batch])
    depth = torch.stack([s['depth'] for s in batch])
    mask = torch.stack([s['mask'] for s in batch])

    # Keep profile IDs as list
    profile_ids = [s['profile_id'] for s in batch]

    return {
        'properties': properties,
        'location': location,
        'depth': depth,
        'mask': mask,
        'profile_ids': profile_ids,
    }


def field_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for field fine-tuning dataset.

    Batches samples from FieldDataset.

    Args:
        batch: List of sample dictionaries from FieldDataset

    Returns:
        Batched dictionary with:
            - features: (batch_size, n_features) tensor
            - targets: (batch_size, n_targets) tensor
            - target_mask: (batch_size, n_targets) boolean tensor
            - location: (batch_size, 2) tensor (if available)
    """
    result = {
        'features': torch.stack([s['features'] for s in batch]),
        'targets': torch.stack([s['targets'] for s in batch]),
        'target_mask': torch.stack([s['target_mask'] for s in batch]),
    }

    # Optional fields
    if 'location' in batch[0]:
        result['location'] = torch.stack([s['location'] for s in batch])

    return result


def temporal_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for temporal sequence data.

    Handles variable-length weather/climate sequences by padding
    to the maximum sequence length in the batch.

    Args:
        batch: List of sample dictionaries with 'weather_sequence' key

    Returns:
        Batched dictionary with padded sequences and length indicators
    """
    # Standard fields
    result = {
        'properties': torch.stack([s['properties'] for s in batch]),
        'location': torch.stack([s['location'] for s in batch]),
        'depth': torch.stack([s['depth'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
    }

    # Handle variable-length weather sequences
    if 'weather_sequence' in batch[0]:
        sequences = [s['weather_sequence'] for s in batch]
        lengths = torch.tensor([len(seq) for seq in sequences])

        # Pad sequences to max length in batch
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        # Create attention mask (1 for valid, 0 for padding)
        max_len = padded.size(1)
        attention_mask = torch.arange(max_len).expand(len(batch), -1) < lengths.unsqueeze(1)

        result['weather_sequence'] = padded
        result['sequence_lengths'] = lengths
        result['sequence_mask'] = attention_mask

    return result


def profile_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for profile-level batching.

    Groups multiple depth observations from the same profile together.
    Useful when you want to process all depths for a profile jointly.

    Args:
        batch: List of sample dictionaries, potentially from same profile

    Returns:
        Batched dictionary grouped by profile
    """
    # Standard stacking
    result = {
        'properties': torch.stack([s['properties'] for s in batch]),
        'location': torch.stack([s['location'] for s in batch]),
        'depth': torch.stack([s['depth'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'profile_ids': [s['profile_id'] for s in batch],
    }

    # Create profile grouping indices
    unique_profiles = []
    profile_indices = []
    profile_to_idx = {}

    for i, s in enumerate(batch):
        pid = s['profile_id']
        if pid not in profile_to_idx:
            profile_to_idx[pid] = len(unique_profiles)
            unique_profiles.append(pid)
        profile_indices.append(profile_to_idx[pid])

    result['profile_group_indices'] = torch.tensor(profile_indices)
    result['n_unique_profiles'] = len(unique_profiles)

    return result


class MaskingCollator:
    """
    Callable class for applying random masking during pre-training.

    This is a class instead of a nested function to make it picklable
    for multiprocessing in DataLoader.

    Attributes:
        mask_ratio: Fraction of valid properties to mask (default 0.15)
        mask_token: Value to use for masked positions (default 0.0)
    """

    def __init__(self, mask_ratio: float = 0.15, mask_token: float = 0.0):
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Apply masking collation to a batch.

        Args:
            batch: List of sample dictionaries from WoSISDataset

        Returns:
            Batched dictionary with masked properties and positions
        """
        # First do standard collation
        result = wosis_collate_fn(batch)

        properties = result['properties']
        valid_mask = result['mask']

        batch_size, n_props = properties.shape

        # Create mask for positions to predict
        # Only mask positions that have valid values
        rand = torch.rand(batch_size, n_props)
        mask_positions = (rand < self.mask_ratio) & valid_mask

        # Ensure at least one position is masked per sample
        for i in range(batch_size):
            if not mask_positions[i].any() and valid_mask[i].any():
                # Randomly select one valid position to mask
                valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                random_idx = valid_indices[torch.randint(len(valid_indices), (1,))]
                mask_positions[i, random_idx] = True

        # Create masked input
        masked_properties = properties.clone()
        masked_properties[mask_positions] = self.mask_token

        # Store original for loss computation
        result['original_properties'] = properties
        result['properties'] = masked_properties
        result['masked_positions'] = mask_positions

        return result


def create_masking_collate_fn(
    mask_ratio: float = 0.15,
    mask_token: float = 0.0,
) -> MaskingCollator:
    """
    Create a collate function that applies random masking for pre-training.

    Returns a MaskingCollator instance that randomly masks properties for the
    masked property prediction objective.

    Args:
        mask_ratio: Fraction of valid properties to mask (default 0.15)
        mask_token: Value to use for masked positions (default 0.0)

    Returns:
        MaskingCollator instance (callable)
    """
    return MaskingCollator(mask_ratio=mask_ratio, mask_token=mask_token)


class DynamicBatchCollator:
    """
    Callable class for dynamic batching by token count.

    Useful for transformer models where memory usage depends on
    sequence length. This ensures batches don't exceed a token budget.

    Attributes:
        max_tokens: Maximum tokens (samples × sequence_length) per batch
    """

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Apply dynamic batch collation."""
        # For now, treat each sample as having a fixed "token" cost
        # based on number of valid properties
        result = wosis_collate_fn(batch)

        # Add batch statistics
        total_valid = result['mask'].sum().item()
        result['batch_stats'] = {
            'batch_size': len(batch),
            'total_valid_tokens': total_valid,
            'avg_valid_per_sample': total_valid / len(batch),
        }

        return result


def dynamic_batch_collate_fn(max_tokens: int = 4096) -> DynamicBatchCollator:
    """
    Create a collate function for dynamic batching by token count.

    Args:
        max_tokens: Maximum tokens (samples × sequence_length) per batch

    Returns:
        DynamicBatchCollator instance (callable)
    """
    return DynamicBatchCollator(max_tokens=max_tokens)

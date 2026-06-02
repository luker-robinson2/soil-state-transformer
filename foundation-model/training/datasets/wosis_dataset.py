"""
WoSIS PyTorch Dataset

Provides PyTorch Dataset classes for loading soil profile data:
- WoSISDataset: Global WoSIS profiles for pre-training
- FieldDataset: Local field data for fine-tuning

Each sample contains:
- Soil properties: [clay, sand, silt, pH, SOC, N, CEC, bdod]
- Location: [latitude, longitude]
- Depth: [upper_depth, lower_depth]
- Mask: Binary mask for valid (non-missing) properties

Supports loading from local paths or GCS URIs (gs://bucket/path).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Optional GCS support
try:
    import gcsfs
    GCSFS_AVAILABLE = True
except ImportError:
    GCSFS_AVAILABLE = False

logger = logging.getLogger(__name__)


def is_gcs_path(path: str) -> bool:
    """Check if path is a GCS URI."""
    return isinstance(path, str) and path.startswith('gs://')


def load_parquet_auto(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load parquet file from local path or GCS URI.

    Args:
        path: Local path or GCS URI (gs://bucket/path/file.parquet)

    Returns:
        Loaded DataFrame
    """
    path_str = str(path)

    if is_gcs_path(path_str):
        if not GCSFS_AVAILABLE:
            raise ImportError(
                "gcsfs is required for loading data from GCS. "
                "Install with: pip install gcsfs"
            )
        fs = gcsfs.GCSFileSystem()
        logger.info(f"Loading data from GCS: {path_str}")
        return pd.read_parquet(path_str, filesystem=fs)
    else:
        return pd.read_parquet(path)


class WoSISDataset(Dataset):
    """
    PyTorch Dataset for WoSIS soil profile pre-training.

    Loads processed WoSIS profiles from parquet files and prepares
    them for the Soil State Transformer. Handles missing values
    through masking.

    Attributes:
        df: Underlying DataFrame with soil observations
        properties: List of property column names
        stats: Normalization statistics (mean, std per property)
    """

    # Default properties matching WoSIS preparation
    DEFAULT_PROPERTIES = (
        'clay',   # Clay content (%)
        'sand',   # Sand content (%)
        'silt',   # Silt content (%)
        'phaq',   # pH in water
        'orgc',   # Organic carbon (g/kg)
        'ntot',   # Total nitrogen (g/kg)
        'cecph7', # CEC at pH 7 (cmol+/kg)
        'bdod',   # Bulk density (kg/dm3)
    )

    # Property aliases for model compatibility
    PROPERTY_ALIASES = {
        'SOC': 'orgc',
        'pH': 'phaq',
        'N': 'ntot',
        'CEC': 'cecph7',
    }

    def __init__(
        self,
        data_path: Union[str, Path],
        properties: Optional[List[str]] = None,
        normalize: bool = True,
        stats: Optional[Dict] = None,
        transform: Optional[callable] = None,
    ):
        """
        Initialize the WoSIS dataset.

        Args:
            data_path: Path to parquet file or DataFrame
            properties: List of property names to include. Uses defaults if None.
            normalize: Whether to normalize properties to zero mean, unit variance
            stats: Pre-computed normalization statistics. Computes from data if None.
            transform: Optional transform to apply to samples
        """
        # Load data (supports local paths and GCS URIs)
        if isinstance(data_path, (str, Path)):
            self.df = load_parquet_auto(data_path)
        elif isinstance(data_path, pd.DataFrame):
            self.df = data_path.copy()
        else:
            raise ValueError(f"Invalid data_path type: {type(data_path)}")

        # Set properties
        self.properties = list(properties or self.DEFAULT_PROPERTIES)

        # Resolve aliases
        self.properties = [
            self.PROPERTY_ALIASES.get(p, p) for p in self.properties
        ]

        # Verify properties exist
        missing = [p for p in self.properties if p not in self.df.columns]
        if missing:
            raise ValueError(f"Properties not found in data: {missing}")

        # Normalization
        self.normalize = normalize
        if normalize:
            if stats is not None:
                self.stats = stats
            else:
                self.stats = self._compute_stats()
        else:
            self.stats = None

        self.transform = transform

        # Precompute tensors for efficiency
        self._precompute_tensors()

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute normalization statistics from data."""
        stats = {}
        for prop in self.properties:
            values = self.df[prop].dropna()
            stats[prop] = {
                'mean': float(values.mean()),
                'std': float(values.std()),
            }
            # Prevent division by zero
            if stats[prop]['std'] < 1e-8:
                stats[prop]['std'] = 1.0
        return stats

    def _precompute_tensors(self):
        """Precompute tensors for faster __getitem__."""
        n_samples = len(self.df)
        n_properties = len(self.properties)

        # Properties tensor
        self._properties = np.zeros((n_samples, n_properties), dtype=np.float32)
        self._masks = np.zeros((n_samples, n_properties), dtype=bool)

        for i, prop in enumerate(self.properties):
            values = self.df[prop].values
            valid = ~np.isnan(values)
            self._masks[:, i] = valid

            if self.normalize and self.stats:
                values = (values - self.stats[prop]['mean']) / self.stats[prop]['std']

            # Replace NaN with 0 (will be masked anyway)
            values = np.nan_to_num(values, 0.0)
            self._properties[:, i] = values

        # Location tensor
        self._locations = np.stack([
            self.df['latitude'].values,
            self.df['longitude'].values,
        ], axis=1).astype(np.float32)

        # Depth tensor
        self._depths = np.stack([
            self.df['upper_depth'].values,
            self.df['lower_depth'].values,
        ], axis=1).astype(np.float32)

        # Profile IDs
        self._profile_ids = self.df['profile_id'].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - properties: (n_properties,) normalized property values
                - location: (2,) latitude, longitude
                - depth: (2,) upper_depth, lower_depth in cm
                - mask: (n_properties,) boolean mask for valid properties
                - profile_id: string identifier
        """
        sample = {
            'properties': torch.from_numpy(self._properties[idx]),
            'location': torch.from_numpy(self._locations[idx]),
            'depth': torch.from_numpy(self._depths[idx]),
            'mask': torch.from_numpy(self._masks[idx]),
            'profile_id': self._profile_ids[idx],
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_normalization_stats(self) -> Dict:
        """Return normalization statistics for use in inference."""
        return self.stats

    def denormalize(
        self,
        properties: torch.Tensor,
        property_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Denormalize property predictions.

        Args:
            properties: Normalized property tensor
            property_indices: Which properties to denormalize. All if None.

        Returns:
            Denormalized tensor
        """
        if not self.normalize or self.stats is None:
            return properties

        if property_indices is None:
            property_indices = list(range(len(self.properties)))

        result = properties.clone()
        for i, prop_idx in enumerate(property_indices):
            prop = self.properties[prop_idx]
            result[..., i] = (
                result[..., i] * self.stats[prop]['std'] +
                self.stats[prop]['mean']
            )
        return result


class FieldDataset(Dataset):
    """
    PyTorch Dataset for field-specific fine-tuning.

    Similar to WoSISDataset but designed for local field data
    with additional features like satellite indices and weather.

    Attributes:
        df: Underlying DataFrame with field observations
        properties: List of property column names
        include_features: Whether to include satellite/weather features
    """

    # Target nutrients for prediction
    DEFAULT_TARGETS = ('K', 'Mg', 'Ca', 'P', 'S', 'Zn', 'Fe', 'Mn', 'Cu', 'B')

    # Input features from satellite/terrain
    DEFAULT_FEATURES = (
        'NDVI', 'NDMI', 'EVI', 'SAVI',  # Vegetation indices
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12',  # Sentinel-2 bands
        'elevation', 'slope', 'aspect', 'twi',  # Terrain
    )

    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        targets: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        stats: Optional[Dict] = None,
    ):
        """
        Initialize field dataset.

        Args:
            data: DataFrame or path to data file
            targets: Target nutrient columns to predict
            features: Feature columns to use as input
            normalize_features: Whether to normalize input features
            normalize_targets: Whether to normalize target values
            stats: Pre-computed normalization statistics
        """
        # Load data (supports local paths and GCS URIs)
        if isinstance(data, (str, Path)):
            data_str = str(data)
            if is_gcs_path(data_str) or data_str.endswith('.parquet'):
                self.df = load_parquet_auto(data)
            else:
                self.df = pd.read_csv(data)
        else:
            self.df = data.copy()

        # Set columns
        self.targets = list(targets or self.DEFAULT_TARGETS)
        self.features = list(features or self.DEFAULT_FEATURES)

        # Filter to available columns
        self.targets = [t for t in self.targets if t in self.df.columns]
        self.features = [f for f in self.features if f in self.df.columns]

        if not self.targets:
            raise ValueError("No target columns found in data")

        # Normalization
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets

        if stats:
            self.stats = stats
        else:
            self.stats = self._compute_stats()

        # Precompute tensors
        self._precompute_tensors()

    def _compute_stats(self) -> Dict:
        """Compute normalization statistics."""
        stats = {'features': {}, 'targets': {}}

        for feat in self.features:
            values = self.df[feat].dropna()
            stats['features'][feat] = {
                'mean': float(values.mean()),
                'std': float(values.std()) if values.std() > 1e-8 else 1.0,
            }

        for target in self.targets:
            values = self.df[target].dropna()
            stats['targets'][target] = {
                'mean': float(values.mean()),
                'std': float(values.std()) if values.std() > 1e-8 else 1.0,
            }

        return stats

    def _precompute_tensors(self):
        """Precompute tensors for efficiency."""
        n_samples = len(self.df)

        # Features
        feature_data = np.zeros((n_samples, len(self.features)), dtype=np.float32)
        for i, feat in enumerate(self.features):
            values = self.df[feat].values.astype(np.float32)
            if self.normalize_features:
                values = (values - self.stats['features'][feat]['mean']) / \
                         self.stats['features'][feat]['std']
            values = np.nan_to_num(values, 0.0)
            feature_data[:, i] = values
        self._features = feature_data

        # Targets
        target_data = np.zeros((n_samples, len(self.targets)), dtype=np.float32)
        target_masks = np.zeros((n_samples, len(self.targets)), dtype=bool)
        for i, target in enumerate(self.targets):
            values = self.df[target].values.astype(np.float32)
            valid = ~np.isnan(values)
            target_masks[:, i] = valid
            if self.normalize_targets:
                values = (values - self.stats['targets'][target]['mean']) / \
                         self.stats['targets'][target]['std']
            values = np.nan_to_num(values, 0.0)
            target_data[:, i] = values
        self._targets = target_data
        self._target_masks = target_masks

        # Location if available
        if 'latitude' in self.df.columns and 'longitude' in self.df.columns:
            self._locations = np.stack([
                self.df['latitude'].values,
                self.df['longitude'].values,
            ], axis=1).astype(np.float32)
        else:
            self._locations = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - features: (n_features,) input features
                - targets: (n_targets,) target values
                - target_mask: (n_targets,) boolean mask for valid targets
                - location: (2,) latitude, longitude (if available)
        """
        sample = {
            'features': torch.from_numpy(self._features[idx]),
            'targets': torch.from_numpy(self._targets[idx]),
            'target_mask': torch.from_numpy(self._target_masks[idx]),
        }

        if self._locations is not None:
            sample['location'] = torch.from_numpy(self._locations[idx])

        return sample

    def denormalize_targets(self, predictions: torch.Tensor) -> torch.Tensor:
        """Denormalize target predictions."""
        if not self.normalize_targets:
            return predictions

        result = predictions.clone()
        for i, target in enumerate(self.targets):
            result[..., i] = (
                result[..., i] * self.stats['targets'][target]['std'] +
                self.stats['targets'][target]['mean']
            )
        return result

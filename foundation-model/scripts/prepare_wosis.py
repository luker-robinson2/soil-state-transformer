"""
WoSIS Data Preparation Script

Processes the WoSIS snapshot into training-ready format:
1. Joins sites, layers, and property tables
2. Filters by depth range and data quality
3. Aggregates to standard depth intervals
4. Splits into train/val/test by spatial groups
5. Saves as parquet files

The WoSIS (World Soil Information Service) snapshot contains ~228k soil profiles
with standardized properties from global sources.

Usage:
    python -m scripts.prepare_wosis --snapshot ./data/raw/wosis/WoSIS_2023_December.zip
    python -m scripts.prepare_wosis --snapshot ./data/raw/wosis/WoSIS_2023_December.zip --output ./data/processed/

Data Structure in ZIP:
    wosis_202312_sites.tsv      - Site locations (217k sites)
    wosis_202312_layers.tsv     - Depth layers (901k layers)
    wosis_202312_profiles.tsv   - Profile metadata (228k profiles)
    wosis_202312_{property}.tsv - Property observations (varying counts)

Properties processed:
    - clay: Clay content (%)
    - sand: Sand content (%)
    - silt: Silt content (%)
    - phaq: pH in water
    - orgc: Organic carbon (g/kg)
    - ntot: Total nitrogen (g/kg)
    - cecph7: CEC at pH 7 (cmol+/kg)
    - bdod: Bulk density (kg/dm3)
"""

import argparse
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WoSISPrepConfig:
    """Configuration for WoSIS data preparation."""

    # Properties to extract (WoSIS column names)
    # Using available properties from WoSIS_2023_December
    properties: Tuple[str, ...] = (
        'clay',    # Clay content (%)
        'sand',    # Sand content (%)
        'silt',    # Silt content (%)
        'phaq',    # pH in water
        'orgc',    # Organic carbon (g/kg)
        'nitkjd',  # Total nitrogen - Kjeldahl (g/kg)
        'cecph7',  # CEC at pH 7 (cmol+/kg)
        'bdfi33',  # Bulk density at field capacity -33kPa (kg/dm3)
    )

    # Standard depth intervals for aggregation (cm)
    depth_intervals: Tuple[Tuple[int, int], ...] = (
        (0, 30),
        (30, 60),
        (60, 100),
        (100, 200),
    )

    # Maximum depth to consider (cm)
    max_depth: int = 200

    # Spatial grid size for cross-validation (degrees)
    grid_size: float = 10.0

    # Train/val/test split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Minimum observations per property to include profile
    min_observations: int = 2

    # Quality filters
    min_quality_score: float = 0.0  # WoSIS quality score threshold

    # Output format
    output_format: str = 'parquet'


class WoSISPreparator:
    """
    Prepares WoSIS snapshot data for foundation model training.

    This class handles the complex joins between WoSIS tables and
    produces training-ready datasets with proper spatial splits.
    """

    def __init__(self, config: Optional[WoSISPrepConfig] = None):
        """
        Initialize the preparator.

        Args:
            config: Configuration for data preparation. Uses defaults if None.
        """
        self.config = config or WoSISPrepConfig()

    def load_wosis_tables(self, snapshot_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all required tables from WoSIS ZIP snapshot.

        Args:
            snapshot_path: Path to WoSIS_2023_December.zip

        Returns:
            Dictionary mapping table names to DataFrames
        """
        tables = {}

        logger.info(f"Loading WoSIS tables from {snapshot_path}")

        with zipfile.ZipFile(snapshot_path, 'r') as zf:
            # List all files
            all_files = zf.namelist()
            tsv_files = [f for f in all_files if f.endswith('.tsv')]
            logger.info(f"Found {len(tsv_files)} TSV files in archive")

            # Load core tables
            core_tables = ['sites', 'layers', 'profiles']
            for table in core_tables:
                matching = [f for f in tsv_files if table in f.lower()]
                if matching:
                    file_name = matching[0]
                    logger.info(f"Loading {table}: {file_name}")
                    with zf.open(file_name) as f:
                        tables[table] = pd.read_csv(f, sep='\t', low_memory=False)
                    logger.info(f"  Loaded {len(tables[table]):,} rows")

            # Load property tables
            for prop in self.config.properties:
                matching = [f for f in tsv_files if f'wosis_202312_{prop}.tsv' in f or f'{prop}.tsv' in f.lower()]
                if matching:
                    file_name = matching[0]
                    logger.info(f"Loading property {prop}: {file_name}")
                    with zf.open(file_name) as f:
                        tables[prop] = pd.read_csv(f, sep='\t', low_memory=False)
                    logger.info(f"  Loaded {len(tables[prop]):,} observations")
                else:
                    logger.warning(f"Property table not found for: {prop}")

        return tables

    def join_tables(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Join sites, layers, and property tables into a single DataFrame.

        The WoSIS structure:
        - sites: site_id, longitude, latitude
        - layers: profile_id, layer_id, site_id, upper_depth, lower_depth
        - properties: profile_id, layer_id, value, longitude, latitude (denormalized)

        We use the property tables directly since they have coordinates.

        Args:
            tables: Dictionary of loaded tables

        Returns:
            Joined DataFrame with all properties
        """
        logger.info("Joining WoSIS tables")

        # Start with layers - has profile_id, layer_id, and depth info
        if 'layers' not in tables:
            raise ValueError("Layers table is required")

        # Get layers with site_id for coordinate lookup
        layers = tables['layers'][['profile_id', 'layer_id', 'site_id', 'upper_depth', 'lower_depth']].copy()

        # Filter by max depth
        layers = layers[layers['lower_depth'] <= self.config.max_depth]
        logger.info(f"Filtered to {len(layers):,} layers within depth range")

        # Get coordinates from sites via site_id
        if 'sites' in tables:
            sites = tables['sites'][['site_id', 'longitude', 'latitude']].drop_duplicates()
            layers = layers.merge(sites, on='site_id', how='left')
            logger.info(f"Added coordinates from sites table")

        # Drop site_id, keep profile_id, layer_id, depths, and coordinates
        df = layers[['profile_id', 'layer_id', 'upper_depth', 'lower_depth', 'longitude', 'latitude']].copy()
        logger.info(f"Base dataframe: {len(df):,} rows")

        # Join each property table
        for prop in self.config.properties:
            if prop not in tables:
                logger.warning(f"Property {prop} not in tables, skipping")
                df[prop] = np.nan
                continue

            prop_df = tables[prop].copy()

            # Property tables have 'value' column
            value_cols = [c for c in prop_df.columns if c == 'value' or 'value' in c.lower()]
            if value_cols:
                value_col = value_cols[0]
            else:
                # Try to find numeric column that's likely the value
                numeric_cols = prop_df.select_dtypes(include=[np.number]).columns
                potential = [c for c in numeric_cols if c not in ['profile_id', 'layer_id', 'upper_depth', 'lower_depth']]
                if potential:
                    value_col = potential[0]
                else:
                    logger.warning(f"Could not identify value column for {prop}")
                    df[prop] = np.nan
                    continue

            # Rename and select columns
            prop_df = prop_df[['profile_id', 'layer_id', value_col]].copy()
            prop_df = prop_df.rename(columns={value_col: prop})

            # Parse value column - WoSIS sometimes has multi-values in curly braces like '{27.5}'
            def parse_value(x):
                if pd.isna(x):
                    return np.nan
                if isinstance(x, (int, float)):
                    return float(x)
                s = str(x).strip()
                # Handle curly brace format: {value} or {val1,val2,...}
                if s.startswith('{') and s.endswith('}'):
                    s = s[1:-1]
                    # If multiple values, take the first one
                    if ',' in s:
                        s = s.split(',')[0]
                try:
                    return float(s)
                except (ValueError, TypeError):
                    return np.nan

            prop_df[prop] = prop_df[prop].apply(parse_value)

            # Handle multiple values per layer (take mean)
            prop_df = prop_df.groupby(['profile_id', 'layer_id'])[prop].mean().reset_index()

            # Join to main dataframe
            df = df.merge(prop_df, on=['profile_id', 'layer_id'], how='left')
            n_valid = df[prop].notna().sum()
            logger.info(f"Added {prop}: {n_valid:,} valid observations ({100*n_valid/len(df):.1f}%)")

        return df

    def aggregate_depths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate observations to standard depth intervals.

        Uses depth-weighted averaging when layers span interval boundaries.

        Args:
            df: DataFrame with layer-level observations

        Returns:
            DataFrame with aggregated depth intervals
        """
        logger.info("Aggregating to standard depth intervals")

        aggregated_rows = []

        for profile_id, group in tqdm(df.groupby('profile_id'), desc="Aggregating profiles"):
            lat = group['latitude'].iloc[0]
            lon = group['longitude'].iloc[0]

            for upper, lower in self.config.depth_intervals:
                # Find overlapping layers
                overlap = group[
                    (group['upper_depth'] < lower) &
                    (group['lower_depth'] > upper)
                ]

                if len(overlap) == 0:
                    continue

                # Calculate overlap weights
                weights = []
                for _, layer in overlap.iterrows():
                    overlap_top = max(layer['upper_depth'], upper)
                    overlap_bottom = min(layer['lower_depth'], lower)
                    overlap_thickness = overlap_bottom - overlap_top
                    weights.append(overlap_thickness)

                weights = np.array(weights)
                weights = weights / weights.sum()

                # Aggregate properties
                row = {
                    'profile_id': profile_id,
                    'latitude': lat,
                    'longitude': lon,
                    'upper_depth': upper,
                    'lower_depth': lower,
                    'depth_interval': f"{upper}-{lower}cm",
                }

                for prop in self.config.properties:
                    values = overlap[prop].values
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 0:
                        # Weighted average of valid values
                        valid_weights = weights[valid_mask]
                        valid_values = values[valid_mask]
                        row[prop] = np.average(valid_values, weights=valid_weights)
                    else:
                        row[prop] = np.nan

                aggregated_rows.append(row)

        result = pd.DataFrame(aggregated_rows)
        logger.info(f"Aggregated to {len(result):,} interval observations")
        return result

    def assign_spatial_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign spatial grid groups for cross-validation.

        Uses a regular grid to ensure spatial separation between
        train, validation, and test sets.

        Args:
            df: DataFrame with latitude/longitude

        Returns:
            DataFrame with 'spatial_group' column
        """
        logger.info(f"Assigning spatial groups (grid size: {self.config.grid_size}°)")

        # Create grid indices
        df = df.copy()
        df['grid_x'] = (df['longitude'] / self.config.grid_size).astype(int)
        df['grid_y'] = (df['latitude'] / self.config.grid_size).astype(int)
        df['spatial_group'] = df['grid_x'].astype(str) + '_' + df['grid_y'].astype(str)

        n_groups = df['spatial_group'].nunique()
        logger.info(f"Created {n_groups} spatial groups")

        # Drop temporary columns
        df = df.drop(columns=['grid_x', 'grid_y'])

        return df

    def filter_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data based on quality criteria.

        Removes profiles with too few valid observations.

        Args:
            df: DataFrame to filter

        Returns:
            Filtered DataFrame
        """
        logger.info("Applying quality filters")
        initial_count = len(df)

        # Count valid properties per row
        property_cols = [p for p in self.config.properties if p in df.columns]
        df['n_valid_properties'] = df[property_cols].notna().sum(axis=1)

        # Filter by minimum observations
        df = df[df['n_valid_properties'] >= self.config.min_observations]
        df = df.drop(columns=['n_valid_properties'])

        final_count = len(df)
        logger.info(f"Filtered from {initial_count:,} to {final_count:,} rows ({100*final_count/initial_count:.1f}%)")

        return df

    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits using spatial groups.

        Ensures entire spatial groups are in one split to prevent
        spatial data leakage.

        Args:
            df: DataFrame with spatial_group column

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating train/val/test splits by spatial groups")

        # Get unique groups and shuffle
        groups = df['spatial_group'].unique()
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(groups)

        # Calculate split points
        n_groups = len(groups)
        train_end = int(n_groups * self.config.train_ratio)
        val_end = train_end + int(n_groups * self.config.val_ratio)

        train_groups = set(groups[:train_end])
        val_groups = set(groups[train_end:val_end])
        test_groups = set(groups[val_end:])

        # Split data
        train_df = df[df['spatial_group'].isin(train_groups)].copy()
        val_df = df[df['spatial_group'].isin(val_groups)].copy()
        test_df = df[df['spatial_group'].isin(test_groups)].copy()

        logger.info(f"Split sizes:")
        logger.info(f"  Train: {len(train_df):,} rows ({len(train_groups)} groups)")
        logger.info(f"  Val:   {len(val_df):,} rows ({len(val_groups)} groups)")
        logger.info(f"  Test:  {len(test_df):,} rows ({len(test_groups)} groups)")

        return train_df, val_df, test_df

    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute dataset statistics for normalization and analysis.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_profiles': df['profile_id'].nunique(),
            'n_observations': len(df),
            'n_spatial_groups': df['spatial_group'].nunique(),
            'properties': {},
            'geographic_bounds': {
                'lat_min': df['latitude'].min(),
                'lat_max': df['latitude'].max(),
                'lon_min': df['longitude'].min(),
                'lon_max': df['longitude'].max(),
            }
        }

        for prop in self.config.properties:
            if prop in df.columns:
                valid = df[prop].dropna()
                stats['properties'][prop] = {
                    'count': len(valid),
                    'coverage': len(valid) / len(df),
                    'mean': valid.mean(),
                    'std': valid.std(),
                    'min': valid.min(),
                    'max': valid.max(),
                    'median': valid.median(),
                    'q25': valid.quantile(0.25),
                    'q75': valid.quantile(0.75),
                }

        return stats

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Save train/val/test splits to output directory.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            output_dir: Directory to save files

        Returns:
            Dictionary mapping split names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if self.config.output_format == 'parquet':
                path = output_dir / f'wosis_{name}.parquet'
                df.to_parquet(path, index=False)
            else:
                path = output_dir / f'wosis_{name}.csv'
                df.to_csv(path, index=False)

            paths[name] = path
            logger.info(f"Saved {name} split to {path}")

        return paths

    def prepare(
        self,
        snapshot_path: Path,
        output_dir: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run the full preparation pipeline.

        Args:
            snapshot_path: Path to WoSIS snapshot ZIP
            output_dir: Optional directory to save processed files

        Returns:
            Tuple of (train_df, val_df, test_df, statistics)
        """
        logger.info("=" * 60)
        logger.info("WoSIS Data Preparation Pipeline")
        logger.info("=" * 60)

        # Load tables
        tables = self.load_wosis_tables(snapshot_path)

        # Join tables
        df = self.join_tables(tables)

        # Aggregate to standard depths
        df = self.aggregate_depths(df)

        # Assign spatial groups
        df = self.assign_spatial_groups(df)

        # Apply quality filters
        df = self.filter_quality(df)

        # Create splits
        train_df, val_df, test_df = self.create_splits(df)

        # Compute statistics
        stats = self.compute_statistics(train_df)
        stats['val_stats'] = self.compute_statistics(val_df)
        stats['test_stats'] = self.compute_statistics(test_df)

        # Save if output directory specified
        if output_dir:
            paths = self.save_splits(train_df, val_df, test_df, output_dir)
            stats['output_paths'] = {k: str(v) for k, v in paths.items()}

            # Save statistics
            import json
            stats_path = output_dir / 'wosis_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            logger.info(f"Saved statistics to {stats_path}")

        logger.info("=" * 60)
        logger.info("Preparation complete!")
        logger.info(f"  Total observations: {len(train_df) + len(val_df) + len(test_df):,}")
        logger.info(f"  Properties: {list(self.config.properties)}")
        logger.info("=" * 60)

        return train_df, val_df, test_df, stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Prepare WoSIS soil profile data for foundation model training'
    )
    parser.add_argument(
        '--snapshot',
        type=str,
        required=True,
        help='Path to WoSIS snapshot ZIP file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/processed/',
        help='Output directory for processed files'
    )
    parser.add_argument(
        '--grid-size',
        type=float,
        default=10.0,
        help='Spatial grid size in degrees (default: 10.0)'
    )
    parser.add_argument(
        '--min-observations',
        type=int,
        default=2,
        help='Minimum valid properties per observation (default: 2)'
    )

    args = parser.parse_args()

    # Configure
    config = WoSISPrepConfig(
        grid_size=args.grid_size,
        min_observations=args.min_observations,
    )

    # Run preparation
    preparator = WoSISPreparator(config)
    train_df, val_df, test_df, stats = preparator.prepare(
        snapshot_path=Path(args.snapshot),
        output_dir=Path(args.output)
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")
    print("\nProperty coverage (train):")
    for prop, prop_stats in stats['properties'].items():
        print(f"  {prop}: {prop_stats['coverage']*100:.1f}% ({prop_stats['count']:,} obs)")


if __name__ == '__main__':
    main()

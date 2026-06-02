"""
Soil State Transformer (SST) - Foundation Model for Soil Property Prediction

This module implements the SST architecture for learning soil state representations
from multi-modal inputs: geographic location, depth, climate history, and base soil properties.

Architecture:
    - Geographic Embedding: Fourier features for spatial encoding
    - Depth Embedding: Sinusoidal + learned for vertical position
    - Soil Property Embedding: Per-property projections with missing value handling
    - Temporal Embedding: Hierarchical climate sequence encoding
    - Transformer Encoder: Multi-head self-attention over modalities
    - Prediction Heads: Property prediction with uncertainty estimation

Usage:
    model = SoilStateTransformer(config)

    # Pre-training
    loss = model.forward_pretrain(batch)

    # Inference
    predictions = model.predict(batch)

    # Fine-tuning with LoRA
    model.enable_lora(r=8, alpha=16)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSTConfig:
    """Configuration for Soil State Transformer."""

    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1

    # Embedding dimensions
    geo_dim: int = 128
    depth_dim: int = 64
    soil_dim: int = 384  # 6 properties × 64
    temporal_dim: int = 160

    # Geographic encoding
    geo_num_freqs: int = 32
    geo_sigma: float = 10.0

    # Temporal encoding
    temporal_seq_len: int = 52  # Weekly for 1 year
    temporal_features: int = 8

    # Soil properties
    soil_properties: Tuple[str, ...] = ('pH', 'clay', 'sand', 'silt', 'CEC', 'BD')
    target_properties: Tuple[str, ...] = ('SOC', 'pH', 'CEC', 'N')

    # Training
    max_depth: float = 200.0  # cm
    mask_probability: float = 0.15


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for geographic coordinates.

    Transforms (lat, lon) into high-dimensional features that capture
    multi-scale spatial patterns, similar to NeRF positional encoding.

    Reference: Tancik et al., "Fourier Features Let Networks Learn
               High Frequency Functions in Low Dimensional Domains"
    """

    def __init__(self, input_dim: int = 2, num_freqs: int = 32, sigma: float = 10.0):
        super().__init__()
        # Random frequency matrix (fixed after initialization)
        B = torch.randn(input_dim, num_freqs) * sigma
        self.register_buffer('B', B)
        # Output: sin and cos for each frequency = num_freqs * 2
        self.output_dim = num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2) tensor of normalized coordinates [lat/90, lon/180]

        Returns:
            (batch, num_freqs * 2) Fourier features
        """
        # Project to frequency space
        proj = 2 * math.pi * x @ self.B  # (batch, num_freqs)
        # Concatenate sin and cos
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DepthEmbedding(nn.Module):
    """
    Embedding for soil depth position.

    Combines sinusoidal positional encoding (for continuity) with
    learned features (for depth-specific patterns).
    """

    def __init__(self, d_model: int = 64, max_depth: float = 200.0):
        super().__init__()
        self.max_depth = max_depth
        self.d_model = d_model

        # Learned projection for depth features
        self.depth_mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Sinusoidal encoding buffer
        pe = self._create_sinusoidal_encoding(int(max_depth) + 1, d_model // 2)
        self.register_buffer('pe', pe)

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, upper_depth: torch.Tensor, lower_depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            upper_depth: (batch,) upper depth in cm
            lower_depth: (batch,) lower depth in cm

        Returns:
            (batch, d_model) depth embedding
        """
        # Compute depth features
        thickness = lower_depth - upper_depth
        center = (upper_depth + lower_depth) / 2

        depth_features = torch.stack([
            upper_depth / self.max_depth,
            lower_depth / self.max_depth,
            thickness / 50.0,
            center / self.max_depth,
            torch.log1p(upper_depth) / math.log1p(self.max_depth),
        ], dim=-1)

        # Learned features
        learned = self.depth_mlp(depth_features)

        # Sinusoidal encoding for center depth
        center_idx = torch.clamp(center.long(), 0, self.pe.size(0) - 1)
        sinusoidal = self.pe[center_idx]

        # Combine
        return learned + F.pad(sinusoidal, (0, self.d_model - sinusoidal.size(-1)))


class SoilPropertyEmbedding(nn.Module):
    """
    Embedding for soil property values with missing value handling.

    Each property gets its own linear projection. Missing values
    are replaced with a learnable embedding.
    """

    def __init__(
        self,
        properties: Tuple[str, ...],
        d_per_property: int = 64,
        property_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        super().__init__()
        self.properties = properties
        self.d_per_property = d_per_property
        self.n_properties = len(properties)
        self.output_dim = self.n_properties * d_per_property

        # Per-property projections
        self.projections = nn.ModuleDict({
            prop: nn.Linear(1, d_per_property)
            for prop in properties
        })

        # Learnable missing value embeddings
        self.missing_embeddings = nn.ParameterDict({
            prop: nn.Parameter(torch.randn(d_per_property) * 0.02)
            for prop in properties
        })

        # Normalization statistics (mean, std per property)
        self.property_stats = property_stats or {}

    def forward(
        self,
        values: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            values: Dict of property name -> (batch,) values
            masks: Dict of property name -> (batch,) binary masks (1=observed)

        Returns:
            (batch, n_properties * d_per_property) concatenated embeddings
        """
        embeddings = []

        for prop in self.properties:
            if prop in values:
                val = values[prop].unsqueeze(-1)  # (batch, 1)

                # Z-score normalization if stats available
                if prop in self.property_stats:
                    mean, std = self.property_stats[prop]
                    val = (val - mean) / (std + 1e-8)

                # Project
                emb = self.projections[prop](val)  # (batch, d)

                # Apply mask if provided
                if masks is not None and prop in masks:
                    mask = masks[prop].unsqueeze(-1)  # (batch, 1)
                    missing = self.missing_embeddings[prop].unsqueeze(0)  # (1, d)
                    emb = mask * emb + (1 - mask) * missing
            else:
                # All missing
                batch_size = next(iter(values.values())).size(0)
                emb = self.missing_embeddings[prop].unsqueeze(0).expand(batch_size, -1)

            embeddings.append(emb)

        return torch.cat(embeddings, dim=-1)


class TemporalEncoder(nn.Module):
    """
    Hierarchical temporal encoder for climate sequences.

    Processes weekly climate data at multiple temporal scales:
    - Weekly: Fine-grained patterns (52 timesteps)
    - Monthly: Medium-term trends (12 timesteps)
    - Seasonal: Cyclic patterns (4 timesteps)
    """

    def __init__(
        self,
        seq_len: int = 52,
        input_features: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(input_features, d_model)

        # Positional encoding for weekly sequence
        self.pos_encoding = self._create_positional_encoding(seq_len, d_model)

        # Transformer encoder for weekly data
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.weekly_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Monthly aggregation (4 weeks per month, 13 months)
        self.monthly_pool = nn.AdaptiveAvgPool1d(12)
        self.monthly_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ),
            num_layers=1
        )

        # Seasonal MLP (4 seasons)
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )

        # Output dimension: weekly + monthly + seasonal
        self.output_dim = d_model + d_model + d_model // 2

    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create learnable positional encoding."""
        return nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) climate sequence

        Returns:
            (batch, output_dim) temporal embedding
        """
        batch_size = x.size(0)

        # Project and add positional encoding
        x = self.input_proj(x)  # (batch, 52, d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Weekly encoding
        weekly_out = self.weekly_encoder(x)  # (batch, 52, d_model)
        weekly_emb = weekly_out[:, 0, :]  # Use first position as summary

        # Monthly aggregation
        monthly_in = self.monthly_pool(weekly_out.transpose(1, 2)).transpose(1, 2)  # (batch, 12, d_model)
        monthly_out = self.monthly_encoder(monthly_in)
        monthly_emb = monthly_out[:, 0, :]

        # Seasonal aggregation (3 months per season)
        seasonal_in = monthly_out.view(batch_size, 4, 3, -1).mean(dim=2)  # (batch, 4, d_model)
        seasonal_emb = self.seasonal_mlp(seasonal_in.view(batch_size, -1))

        return torch.cat([weekly_emb, monthly_emb, seasonal_emb], dim=-1)


class SoilStateTransformer(nn.Module):
    """
    Soil State Transformer (SST) - Foundation Model.

    Combines multi-modal inputs through a transformer encoder to produce
    a unified soil state representation for property prediction.
    """

    def __init__(self, config: SSTConfig):
        super().__init__()
        self.config = config

        # Embedding modules
        self.geo_embedding = FourierFeatures(
            input_dim=2,
            num_freqs=config.geo_num_freqs,
            sigma=config.geo_sigma
        )
        self.geo_proj = nn.Linear(self.geo_embedding.output_dim, config.d_model)

        self.depth_embedding = DepthEmbedding(
            d_model=config.depth_dim,
            max_depth=config.max_depth
        )
        self.depth_proj = nn.Linear(config.depth_dim, config.d_model)

        self.soil_embedding = SoilPropertyEmbedding(
            properties=config.soil_properties,
            d_per_property=config.soil_dim // len(config.soil_properties)
        )
        self.soil_proj = nn.Linear(self.soil_embedding.output_dim, config.d_model)

        self.temporal_encoder = TemporalEncoder(
            seq_len=config.temporal_seq_len,
            input_features=config.temporal_features,
            d_model=config.temporal_dim // 2,
            n_heads=4,
            n_layers=2,
            dropout=config.dropout
        )
        self.temporal_proj = nn.Linear(self.temporal_encoder.output_dim, config.d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Prediction heads
        self.prediction_heads = nn.ModuleDict({
            prop: self._create_prediction_head(config.d_model)
            for prop in config.target_properties
        })

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small values for stability."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _create_prediction_head(self, input_dim: int) -> nn.Module:
        """Create a prediction head with uncertainty estimation."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 2),  # mean and log_var
        )

    def encode(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        upper_depth: torch.Tensor,
        lower_depth: torch.Tensor,
        soil_properties: Dict[str, torch.Tensor],
        climate_sequence: torch.Tensor,
        soil_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Encode inputs into soil state embedding.

        Args:
            lat: (batch,) latitude in degrees
            lon: (batch,) longitude in degrees
            upper_depth: (batch,) upper depth in cm
            lower_depth: (batch,) lower depth in cm
            soil_properties: Dict of property -> (batch,) values
            climate_sequence: (batch, seq_len, features) climate data
            soil_masks: Optional masks for missing soil properties

        Returns:
            (batch, d_model) soil state embedding
        """
        batch_size = lat.size(0)

        # Geographic embedding
        geo_input = torch.stack([lat / 90.0, lon / 180.0], dim=-1)
        geo_emb = self.geo_proj(self.geo_embedding(geo_input))

        # Depth embedding
        depth_emb = self.depth_proj(self.depth_embedding(upper_depth, lower_depth))

        # Soil property embedding
        soil_emb = self.soil_proj(self.soil_embedding(soil_properties, soil_masks))

        # Temporal embedding
        temporal_emb = self.temporal_proj(self.temporal_encoder(climate_sequence))

        # Combine into sequence: [CLS, geo, depth, soil, temporal]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.stack([geo_emb, depth_emb, soil_emb, temporal_emb], dim=1)
        sequence = torch.cat([cls_tokens, sequence], dim=1)

        # Transformer encoding
        encoded = self.encoder(sequence)

        # Return CLS token embedding
        return encoded[:, 0, :]

    def predict(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        upper_depth: torch.Tensor,
        lower_depth: torch.Tensor,
        soil_properties: Dict[str, torch.Tensor],
        climate_sequence: torch.Tensor,
        soil_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Predict soil properties with uncertainty.

        Returns:
            Dict of property -> {'mean': tensor, 'std': tensor}
        """
        # Get embedding
        h = self.encode(
            lat, lon, upper_depth, lower_depth,
            soil_properties, climate_sequence, soil_masks
        )

        # Predictions for each target property
        predictions = {}
        for prop, head in self.prediction_heads.items():
            output = head(h)  # (batch, 2)
            mean = output[:, 0]
            log_var = output[:, 1]
            std = F.softplus(log_var).sqrt()
            predictions[prop] = {'mean': mean, 'std': std}

        return predictions

    def forward_pretrain(
        self,
        lat: torch.Tensor,
        lon: torch.Tensor,
        upper_depth: torch.Tensor,
        lower_depth: torch.Tensor,
        soil_properties: Dict[str, torch.Tensor],
        climate_sequence: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-training forward pass with masked property prediction.

        Args:
            ... (same as encode)
            targets: Dict of property -> (batch,) ground truth values

        Returns:
            Dict containing 'loss' and per-property losses
        """
        # Create random masks for input properties
        soil_masks = {}
        for prop in soil_properties:
            mask = (torch.rand(lat.size(0), device=lat.device) > self.config.mask_probability).float()
            soil_masks[prop] = mask

        # Get predictions
        predictions = self.predict(
            lat, lon, upper_depth, lower_depth,
            soil_properties, climate_sequence, soil_masks
        )

        # Compute losses
        losses = {}
        total_loss = 0.0

        for prop, pred in predictions.items():
            if prop in targets:
                target = targets[prop]
                mean = pred['mean']
                std = pred['std']

                # Negative log-likelihood loss
                nll = 0.5 * (torch.log(std ** 2 + 1e-8) + ((target - mean) ** 2) / (std ** 2 + 1e-8))
                loss = nll.mean()

                losses[f'loss_{prop}'] = loss
                total_loss = total_loss + loss

        losses['loss'] = total_loss
        return losses


def create_sst_model(config: Optional[SSTConfig] = None) -> SoilStateTransformer:
    """Factory function to create SST model."""
    if config is None:
        config = SSTConfig()
    return SoilStateTransformer(config)


# Example usage
if __name__ == '__main__':
    # Create model
    config = SSTConfig(
        d_model=256,  # Smaller for testing
        n_layers=2,
        n_heads=4,
    )
    model = create_sst_model(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4

    lat = torch.randn(batch_size) * 30 + 40  # ~40°N ± 30°
    lon = torch.randn(batch_size) * 50 - 100  # ~100°W ± 50°
    upper_depth = torch.randint(0, 30, (batch_size,)).float()
    lower_depth = upper_depth + torch.randint(10, 30, (batch_size,)).float()

    soil_properties = {
        'pH': torch.randn(batch_size) * 1.5 + 6.5,
        'clay': torch.rand(batch_size) * 50 + 10,
        'sand': torch.rand(batch_size) * 50 + 20,
        'silt': torch.rand(batch_size) * 40 + 20,
        'CEC': torch.rand(batch_size) * 30 + 5,
        'BD': torch.rand(batch_size) * 0.5 + 1.2,
    }

    climate_sequence = torch.randn(batch_size, 52, 8)

    # Forward pass
    predictions = model.predict(
        lat, lon, upper_depth, lower_depth,
        soil_properties, climate_sequence
    )

    print("\nPredictions:")
    for prop, pred in predictions.items():
        print(f"  {prop}: mean={pred['mean'].mean():.3f}, std={pred['std'].mean():.3f}")

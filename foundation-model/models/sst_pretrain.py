"""
Simplified SST for WoSIS Pre-training

A lightweight version of the Soil State Transformer optimized for
pre-training on WoSIS data using masked property prediction.

This variant works with:
- Location (lat, lon)
- Depth (upper, lower)
- Soil properties as a tensor

Climate data is not used during pre-training. The full SST with temporal
encoding is used during fine-tuning when climate data is available.

Usage:
    >>> from models.sst_pretrain import SSTPretrainModel
    >>> model = SSTPretrainModel(n_properties=8)
    >>> predictions, log_var = model.forward_pretrain(props, loc, depth, mask)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSTPretrainConfig:
    """Configuration for pre-training SST."""

    # Model dimensions
    embedding_dim: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1

    # Input dimensions
    n_properties: int = 8

    # Geographic encoding
    geo_num_freqs: int = 32
    geo_sigma: float = 10.0

    # Depth encoding
    max_depth: float = 200.0
    depth_dim: int = 64

    # Prediction
    predict_uncertainty: bool = True


class FourierFeatures(nn.Module):
    """Fourier feature encoding for geographic coordinates."""

    def __init__(self, input_dim: int = 2, num_freqs: int = 32, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(input_dim, num_freqs) * sigma
        self.register_buffer('B', B)
        self.output_dim = num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class DepthEncoder(nn.Module):
    """Encode depth interval to embedding."""

    def __init__(self, d_model: int = 64, max_depth: float = 200.0):
        super().__init__()
        self.max_depth = max_depth

        # Learned projection for depth features
        self.mlp = nn.Sequential(
            nn.Linear(5, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, upper_depth: torch.Tensor, lower_depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            upper_depth: (batch,) upper depth in cm
            lower_depth: (batch,) lower depth in cm
        Returns:
            (batch, d_model) depth embedding
        """
        thickness = lower_depth - upper_depth
        center = (upper_depth + lower_depth) / 2

        # Normalize features
        features = torch.stack([
            upper_depth / self.max_depth,
            lower_depth / self.max_depth,
            thickness / self.max_depth,
            center / self.max_depth,
            torch.log1p(center) / math.log(self.max_depth + 1),
        ], dim=-1)

        return self.mlp(features)


class PropertyEncoder(nn.Module):
    """Encode soil properties to embeddings."""

    def __init__(self, n_properties: int, d_model: int):
        super().__init__()
        self.n_properties = n_properties

        # Per-property embedding with value encoding
        self.property_embeddings = nn.Embedding(n_properties, d_model)
        self.value_proj = nn.Linear(1, d_model)

        # Combine property ID and value
        self.combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Mask token for masked properties
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(
        self,
        properties: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            properties: (batch, n_properties) property values
            property_mask: (batch, n_properties) True where property is visible

        Returns:
            (batch, n_properties, d_model) property embeddings
        """
        batch_size = properties.size(0)
        device = properties.device

        # Get property type embeddings
        prop_ids = torch.arange(self.n_properties, device=device)
        prop_emb = self.property_embeddings(prop_ids)  # (n_props, d_model)
        prop_emb = prop_emb.unsqueeze(0).expand(batch_size, -1, -1)

        # Get value embeddings
        value_emb = self.value_proj(properties.unsqueeze(-1))  # (batch, n_props, d_model)

        # Combine
        combined = torch.cat([prop_emb, value_emb], dim=-1)  # (batch, n_props, d_model*2)
        output = self.combine(combined)  # (batch, n_props, d_model)

        # Apply mask (replace masked positions with mask token)
        if property_mask is not None:
            mask_expanded = property_mask.unsqueeze(-1)  # (batch, n_props, 1)
            mask_token_expanded = self.mask_token.view(1, 1, -1).expand_as(output)
            output = torch.where(mask_expanded, output, mask_token_expanded)

        return output


class SSTPretrainModel(nn.Module):
    """
    Simplified Soil State Transformer for pre-training.

    Uses only location, depth, and soil properties (no climate data).
    Pre-trains via masked property prediction.
    """

    def __init__(
        self,
        n_properties: int = 8,
        embedding_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        predict_uncertainty: bool = True,
    ):
        super().__init__()

        self.n_properties = n_properties
        self.embedding_dim = embedding_dim
        self.predict_uncertainty = predict_uncertainty

        # Geographic encoding
        self.geo_encoder = FourierFeatures(input_dim=2, num_freqs=32, sigma=10.0)
        self.geo_proj = nn.Linear(self.geo_encoder.output_dim, embedding_dim)

        # Depth encoding
        self.depth_encoder = DepthEncoder(d_model=embedding_dim // 2)
        self.depth_proj = nn.Linear(embedding_dim // 2, embedding_dim)

        # Property encoding
        self.property_encoder = PropertyEncoder(n_properties, embedding_dim)

        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)

        # Positional encoding for sequence (CLS, geo, depth, prop1, prop2, ...)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 3 + n_properties, embedding_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection for each property
        # Predicts the original value (and optionally log variance)
        out_dim = 2 if predict_uncertainty else 1
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
                nn.Linear(embedding_dim // 2, out_dim),
            )
            for _ in range(n_properties)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(
        self,
        properties: torch.Tensor,
        location: torch.Tensor,
        depth: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode inputs to embeddings.

        Args:
            properties: (batch, n_properties) soil property values (normalized)
            location: (batch, 2) [lat, lon] in degrees
            depth: (batch, 2) [upper_depth, lower_depth] in cm
            property_mask: (batch, n_properties) True where visible

        Returns:
            (batch, embedding_dim) aggregated embedding
            (batch, n_properties, embedding_dim) per-property embeddings
        """
        batch_size = properties.size(0)

        # Geographic embedding
        geo_input = torch.stack([
            location[:, 0] / 90.0,   # Normalize lat to [-1, 1]
            location[:, 1] / 180.0,  # Normalize lon to [-1, 1]
        ], dim=-1)
        geo_emb = self.geo_proj(self.geo_encoder(geo_input))  # (batch, emb_dim)

        # Depth embedding
        depth_emb = self.depth_proj(
            self.depth_encoder(depth[:, 0], depth[:, 1])
        )  # (batch, emb_dim)

        # Property embeddings
        prop_emb = self.property_encoder(properties, property_mask)  # (batch, n_props, emb_dim)

        # Build sequence: [CLS, geo, depth, prop1, prop2, ...]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        geo_emb = geo_emb.unsqueeze(1)
        depth_emb = depth_emb.unsqueeze(1)

        sequence = torch.cat([cls_tokens, geo_emb, depth_emb, prop_emb], dim=1)

        # Add positional encoding
        sequence = sequence + self.pos_embedding

        # Transformer encoding
        encoded = self.encoder(sequence)

        # Return CLS embedding and per-property embeddings
        cls_output = encoded[:, 0, :]
        prop_outputs = encoded[:, 3:, :]  # Skip CLS, geo, depth

        return cls_output, prop_outputs

    def forward_pretrain(
        self,
        properties: torch.Tensor,
        location: torch.Tensor,
        depth: torch.Tensor,
        property_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for pre-training.

        Args:
            properties: (batch, n_properties) masked property values
            location: (batch, 2) [lat, lon]
            depth: (batch, 2) [upper, lower]
            property_mask: (batch, n_properties) True where visible

        Returns:
            predictions: (batch, n_properties) predicted values
            log_var: (batch, n_properties) log variance (if predict_uncertainty)
        """
        # Encode
        cls_emb, prop_emb = self.encode(properties, location, depth, property_mask)

        # Predict each property from its embedding
        predictions = []
        log_vars = []

        for i, head in enumerate(self.output_heads):
            out = head(prop_emb[:, i, :])  # (batch, 1 or 2)

            if self.predict_uncertainty:
                predictions.append(out[:, 0])
                log_vars.append(out[:, 1])
            else:
                predictions.append(out[:, 0])

        predictions = torch.stack(predictions, dim=1)  # (batch, n_props)

        if self.predict_uncertainty:
            log_var = torch.stack(log_vars, dim=1)  # (batch, n_props)
            return predictions, log_var
        else:
            return predictions, None

    def predict(
        self,
        properties: torch.Tensor,
        location: torch.Tensor,
        depth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict soil properties (all visible, no masking).

        Returns:
            predictions: (batch, n_properties)
            uncertainties: (batch, n_properties) standard deviations
        """
        mask = torch.ones_like(properties, dtype=torch.bool)
        predictions, log_var = self.forward_pretrain(properties, location, depth, mask)

        if log_var is not None:
            std = torch.exp(0.5 * log_var)
        else:
            std = torch.zeros_like(predictions)

        return predictions, std

    def save(self, path: str):
        """Save model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'n_properties': self.n_properties,
            'embedding_dim': self.embedding_dim,
            'predict_uncertainty': self.predict_uncertainty,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'SSTPretrainModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            n_properties=checkpoint['n_properties'],
            embedding_dim=checkpoint['embedding_dim'],
            predict_uncertainty=checkpoint['predict_uncertainty'],
        )
        model.load_state_dict(checkpoint['state_dict'])

        return model


# Test
if __name__ == '__main__':
    # Create model
    model = SSTPretrainModel(
        n_properties=8,
        embedding_dim=256,
        n_heads=8,
        n_layers=6,
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test forward pass
    batch_size = 16
    properties = torch.randn(batch_size, 8)
    location = torch.randn(batch_size, 2) * torch.tensor([30, 50])  # lat, lon
    depth = torch.tensor([[0, 30]] * batch_size, dtype=torch.float32)
    mask = torch.rand(batch_size, 8) > 0.15  # ~85% visible

    predictions, log_var = model.forward_pretrain(properties, location, depth, mask)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Log variance shape: {log_var.shape}")
    print("Forward pass successful!")

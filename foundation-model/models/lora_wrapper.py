"""
LoRA Wrapper for Soil State Transformer

Provides Low-Rank Adaptation (LoRA) functionality for parameter-efficient
fine-tuning of the pre-trained SST model.

LoRA works by adding trainable low-rank decomposition matrices to
frozen pre-trained weights:

    W' = W + BA

Where:
    - W: Original frozen weights (d × k)
    - B: Low-rank matrix (d × r), initialized to zeros
    - A: Low-rank matrix (r × k), initialized with Kaiming uniform
    - r: Rank (typically 4-16), r << min(d, k)

This approach:
    - Reduces trainable parameters by ~99%
    - Prevents catastrophic forgetting of pre-trained knowledge
    - Enables efficient multi-field deployment (one base model, many adapters)

Supports both:
    - Custom LoRA implementation (no dependencies)
    - PEFT library integration (if available)

Usage:
    >>> from models.lora_wrapper import apply_lora_to_sst, save_lora_adapter
    >>> from training.config import LoRAConfig
    >>>
    >>> # Load pre-trained model
    >>> model = SoilStateTransformer.load('checkpoints/pretrain/best_model.pt')
    >>>
    >>> # Apply LoRA
    >>> config = LoRAConfig(rank=8, alpha=16)
    >>> lora_model = apply_lora_to_sst(model, config)
    >>>
    >>> # Fine-tune...
    >>>
    >>> # Save adapter
    >>> save_lora_adapter(lora_model, 'adapters/field_001/')
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional PEFT integration
try:
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model, PeftModel, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for linear transformations.

    Wraps an existing Linear layer and adds trainable low-rank matrices
    while keeping the original weights frozen.

    Forward pass:
        output = W @ x + (B @ A) @ x * scaling

    Attributes:
        original_layer: Frozen original Linear layer
        lora_A: Low-rank matrix A (rank × in_features)
        lora_B: Low-rank matrix B (out_features × rank)
        scaling: Scaling factor (alpha / rank)
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        """
        Initialize LoRA layer.

        Args:
            original_layer: The nn.Linear layer to wrap
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA weights
            dropout: Dropout probability for LoRA path
        """
        super().__init__()

        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices
        # A: Initialized with Kaiming uniform (same as PyTorch Linear default)
        # B: Initialized to zeros (ensures LoRA has no effect initially)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zeros so initial output = original output

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Track if merged
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            x: Input tensor (*, in_features)

        Returns:
            Output tensor (*, out_features)
        """
        if self.merged:
            # Weights have been merged, use original layer
            return self.original_layer(x)

        # Original path (frozen)
        original_output = self.original_layer(x)

        # LoRA path (trainable)
        # x @ A.T @ B.T = (B @ A @ x.T).T
        lora_output = self.dropout(x)
        lora_output = lora_output @ self.lora_A.T  # (*, rank)
        lora_output = lora_output @ self.lora_B.T  # (*, out_features)

        return original_output + self.scaling * lora_output

    def merge_weights(self):
        """Merge LoRA weights into the original layer for inference efficiency."""
        if self.merged:
            return

        # W' = W + scaling * B @ A
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.original_layer.weight.data += delta

        self.merged = True
        logger.debug(f"Merged LoRA weights (delta norm: {delta.norm():.4f})")

    def unmerge_weights(self):
        """Unmerge LoRA weights (restore original weights)."""
        if not self.merged:
            return

        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.original_layer.weight.data -= delta

        self.merged = False


class LoRAWrapper(nn.Module):
    """
    Wrapper that applies LoRA to a model's attention layers.

    Attributes:
        base_model: Original model with frozen weights
        lora_layers: Dictionary of LoRA layers by name
    """

    def __init__(
        self,
        base_model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Initialize LoRA wrapper.

        Args:
            base_model: Model to apply LoRA to
            rank: LoRA rank
            alpha: LoRA scaling factor
            dropout: LoRA dropout
            target_modules: List of module name patterns to apply LoRA to
        """
        super().__init__()

        self.base_model = base_model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        # Default target modules for SST
        if target_modules is None:
            target_modules = ['query_proj', 'key_proj', 'value_proj', 'output_proj']

        self.target_modules = target_modules
        self.lora_layers: Dict[str, LoRALayer] = {}

        # Apply LoRA to target modules
        self._apply_lora()

        # Log parameter counts
        self._log_parameters()

    def _apply_lora(self):
        """Replace target Linear layers with LoRA layers."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module matches any target pattern
                if any(target in name for target in self.target_modules):
                    # Get parent module and attribute name
                    parent_name, attr_name = name.rsplit('.', 1) if '.' in name else ('', name)
                    parent = self._get_module_by_name(parent_name) if parent_name else self.base_model

                    # Create LoRA layer
                    lora_layer = LoRALayer(
                        original_layer=module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                    )

                    # Replace in parent
                    setattr(parent, attr_name, lora_layer)
                    self.lora_layers[name] = lora_layer

                    logger.debug(f"Applied LoRA to: {name}")

        logger.info(f"Applied LoRA to {len(self.lora_layers)} layers")

    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get a module by its dotted name."""
        module = self.base_model
        for attr in name.split('.'):
            module = getattr(module, attr)
        return module

    def _log_parameters(self):
        """Log parameter statistics."""
        total_params = sum(p.numel() for p in self.base_model.parameters())
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        lora_params = sum(
            layer.lora_A.numel() + layer.lora_B.numel()
            for layer in self.lora_layers.values()
        )

        logger.info(f"Parameter counts:")
        logger.info(f"  Total:     {total_params:,}")
        logger.info(f"  LoRA:      {lora_params:,}")
        logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        return self.base_model(*args, **kwargs)

    def merge_and_unload(self) -> nn.Module:
        """Merge LoRA weights and return the base model."""
        for layer in self.lora_layers.values():
            layer.merge_weights()
        return self.base_model

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only LoRA parameters."""
        state_dict = {}
        for name, layer in self.lora_layers.items():
            state_dict[f'{name}.lora_A'] = layer.lora_A
            state_dict[f'{name}.lora_B'] = layer.lora_B
        return state_dict

    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA parameters from state dict."""
        for name, layer in self.lora_layers.items():
            if f'{name}.lora_A' in state_dict:
                layer.lora_A.data = state_dict[f'{name}.lora_A']
            if f'{name}.lora_B' in state_dict:
                layer.lora_B.data = state_dict[f'{name}.lora_B']


def apply_lora_to_sst(
    model: nn.Module,
    config,  # LoRAConfig from training.config
    use_peft: bool = True,
) -> Union[LoRAWrapper, 'PeftModel']:
    """
    Apply LoRA adapters to Soil State Transformer attention layers.

    Args:
        model: Pre-trained SST model
        config: LoRAConfig with LoRA parameters
        use_peft: Whether to use PEFT library (if available)

    Returns:
        Model wrapped with LoRA adapters
    """
    if use_peft and PEFT_AVAILABLE:
        return _apply_peft_lora(model, config)
    else:
        return _apply_custom_lora(model, config)


def _apply_peft_lora(model: nn.Module, config) -> 'PeftModel':
    """Apply LoRA using PEFT library."""
    peft_config = PeftLoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias=config.bias,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_model = get_peft_model(model, peft_config)

    # Log parameter counts
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(f"PEFT LoRA applied: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    return peft_model


def _apply_custom_lora(model: nn.Module, config) -> LoRAWrapper:
    """Apply LoRA using custom implementation."""
    return LoRAWrapper(
        base_model=model,
        rank=config.rank,
        alpha=config.alpha,
        dropout=config.dropout,
        target_modules=list(config.target_modules),
    )


def merge_lora_weights(model: Union[LoRAWrapper, 'PeftModel']) -> nn.Module:
    """
    Merge LoRA weights into base model for efficient inference.

    After merging, the model no longer has separate LoRA weights
    and behaves like a standard model (but with modified weights).

    Args:
        model: LoRA-wrapped model

    Returns:
        Base model with merged weights
    """
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        return model.merge_and_unload()
    elif isinstance(model, LoRAWrapper):
        return model.merge_and_unload()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def save_lora_adapter(
    model: Union[LoRAWrapper, 'PeftModel'],
    path: Union[str, Path],
    include_config: bool = True,
):
    """
    Save only LoRA adapter weights.

    Args:
        model: LoRA-wrapped model
        path: Directory to save adapter
        include_config: Whether to save configuration
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        model.save_pretrained(path)
        logger.info(f"Saved PEFT adapter to {path}")
    elif isinstance(model, LoRAWrapper):
        # Save LoRA state dict
        state_dict = model.get_lora_state_dict()
        torch.save(state_dict, path / 'lora_weights.pt')

        # Save config
        if include_config:
            import json
            config = {
                'rank': model.rank,
                'alpha': model.alpha,
                'dropout': model.dropout,
                'target_modules': model.target_modules,
            }
            with open(path / 'lora_config.json', 'w') as f:
                json.dump(config, f, indent=2)

        logger.info(f"Saved custom LoRA adapter to {path}")
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def load_lora_adapter(
    base_model: nn.Module,
    adapter_path: Union[str, Path],
) -> Union[LoRAWrapper, 'PeftModel']:
    """
    Load LoRA adapter onto a base model.

    Args:
        base_model: Pre-trained base model
        adapter_path: Path to saved adapter

    Returns:
        Model with loaded LoRA adapter
    """
    adapter_path = Path(adapter_path)

    if PEFT_AVAILABLE and (adapter_path / 'adapter_config.json').exists():
        # PEFT adapter
        return PeftModel.from_pretrained(base_model, adapter_path)
    elif (adapter_path / 'lora_weights.pt').exists():
        # Custom adapter
        import json

        # Load config
        with open(adapter_path / 'lora_config.json') as f:
            config = json.load(f)

        # Create wrapper
        wrapper = LoRAWrapper(
            base_model=base_model,
            rank=config['rank'],
            alpha=config['alpha'],
            dropout=config['dropout'],
            target_modules=config['target_modules'],
        )

        # Load weights
        state_dict = torch.load(adapter_path / 'lora_weights.pt')
        wrapper.load_lora_state_dict(state_dict)

        logger.info(f"Loaded custom LoRA adapter from {adapter_path}")
        return wrapper
    else:
        raise ValueError(f"No valid adapter found at {adapter_path}")


def get_lora_info(model: Union[LoRAWrapper, 'PeftModel']) -> Dict:
    """
    Get information about LoRA configuration and parameters.

    Args:
        model: LoRA-wrapped model

    Returns:
        Dictionary with LoRA information
    """
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        config = model.peft_config['default']
        return {
            'type': 'peft',
            'rank': config.r,
            'alpha': config.lora_alpha,
            'dropout': config.lora_dropout,
            'target_modules': config.target_modules,
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    elif isinstance(model, LoRAWrapper):
        return {
            'type': 'custom',
            'rank': model.rank,
            'alpha': model.alpha,
            'dropout': model.dropout,
            'target_modules': model.target_modules,
            'n_lora_layers': len(model.lora_layers),
            'trainable_params': sum(p.numel() for p in model.base_model.parameters() if p.requires_grad),
        }
    else:
        return {'type': 'unknown'}

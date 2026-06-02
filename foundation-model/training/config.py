"""
Training Configuration Classes

Provides dataclass configurations for:
- PretrainConfig: Global pre-training on WoSIS data
- LoRAConfig: Low-rank adaptation parameters
- FinetuneConfig: Field-specific fine-tuning
- InferenceConfig: Model inference settings

These configurations support:
- YAML serialization/deserialization
- CLI argument parsing
- Default values with validation
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import yaml


@dataclass
class ModelConfig:
    """
    Soil State Transformer architecture configuration.

    Defines the model dimensions and structure.
    """
    # Embedding dimensions
    embedding_dim: int = 256
    hidden_dim: int = 512

    # Transformer architecture
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    # Input specifications
    n_properties: int = 8  # SOC, pH, clay, sand, silt, N, CEC, bdod
    n_depth_intervals: int = 4  # 0-30, 30-60, 60-100, 100-200

    # Position encoding
    max_depth: int = 200  # Maximum depth in cm
    use_fourier_features: bool = True
    fourier_features_dim: int = 32

    # Prediction heads
    predict_uncertainty: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PretrainConfig:
    """
    Configuration for SST pre-training on WoSIS global data.

    Pre-training uses masked property prediction: randomly mask
    15% of valid soil properties and train the model to predict
    the masked values from context.
    """
    # ===== Data Paths =====
    train_path: str = 'data/processed/wosis_train.parquet'
    val_path: str = 'data/processed/wosis_val.parquet'
    output_dir: str = 'checkpoints/pretrain/'

    # ===== Model Configuration =====
    model: ModelConfig = field(default_factory=ModelConfig)

    # ===== Properties =====
    # WoSIS column names:
    # nitkjd = Total nitrogen (Kjeldahl method)
    # bdfi33 = Bulk density at field capacity (-33kPa)
    properties: Tuple[str, ...] = (
        'clay', 'sand', 'silt', 'phaq', 'orgc', 'nitkjd', 'cecph7', 'bdfi33'
    )

    # ===== Training Hyperparameters =====
    # Batch size and accumulation
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 1024  # batch_size × gradient_accumulation

    # Learning rate schedule
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 50000

    # LR scheduler
    lr_scheduler: str = 'cosine'  # 'cosine', 'linear', 'constant'
    min_lr_ratio: float = 0.1  # Minimum LR as fraction of initial

    # ===== Masked Pre-training =====
    mask_ratio: float = 0.15  # Fraction of properties to mask
    mask_token_value: float = 0.0  # Value for masked positions

    # ===== Regularization =====
    label_smoothing: float = 0.0
    gradient_clip_norm: float = 1.0

    # ===== Infrastructure =====
    device: str = 'mps'  # 'cuda', 'cpu', 'mps' - default to MPS for Apple Silicon
    mixed_precision: bool = True  # Use FP16/BF16
    precision: str = 'fp16'  # 'fp16', 'bf16', 'fp32'
    num_workers: int = 4
    pin_memory: bool = True

    # ===== Checkpointing =====
    checkpoint_every: int = 5000
    keep_last_n_checkpoints: int = 3
    save_best_only: bool = True

    # ===== Logging =====
    log_every: int = 100
    eval_every: int = 1000

    # WandB settings
    use_wandb: bool = True
    wandb_project: str = 'soil-state-transformer'
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ['pretrain', 'wosis'])

    # ===== Reproducibility =====
    seed: int = 42

    # ===== Cloud Training (GCS & Vertex AI) =====
    use_gcs: bool = False  # Enable GCS for checkpoints and data
    gcs_bucket: str = 'your-research-bucket'
    gcs_checkpoint_prefix: str = 'checkpoints/pretrain/'
    gcs_data_prefix: str = 'data/processed/'

    # Vertex AI TensorBoard integration
    vertex_tensorboard: Optional[str] = None  # TensorBoard instance ID

    # Auto-detect cloud environment
    auto_detect_device: bool = True  # Auto-detect CUDA/TPU in cloud

    def __post_init__(self):
        """Validate configuration."""
        import os
        import torch

        # Compute effective batch size
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

        # Validate mask ratio
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be between 0 and 1, got {self.mask_ratio}")

        # Auto-detect device in cloud environments
        if self.auto_detect_device:
            # Check for Vertex AI environment
            if os.environ.get('AIP_TRAINING_DATA_URI'):
                # Running in Vertex AI
                if torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

        # Create output directory only for local paths
        if not self.output_dir.startswith('gs://') and not self.use_gcs:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['model'] = self.model.to_dict()
        return result

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'PretrainConfig':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        if 'model' in data:
            data['model'] = ModelConfig(**data['model'])

        return cls(**data)

    def to_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class LoRAConfig:
    """
    Configuration for Low-Rank Adaptation (LoRA).

    LoRA adds trainable low-rank matrices to frozen pre-trained weights,
    enabling parameter-efficient fine-tuning.

    Mathematical formulation:
        W' = W + BA
        where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)

    Attributes:
        rank: Rank of the low-rank matrices (r in the formula)
        alpha: Scaling factor for LoRA weights
        dropout: Dropout applied to LoRA layers
        target_modules: Which layers to apply LoRA to
    """
    # ===== LoRA Parameters =====
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    # Target modules to apply LoRA
    # These should match the attention projection layer names in SST
    target_modules: Tuple[str, ...] = (
        'query_proj',
        'key_proj',
        'value_proj',
        'output_proj',
    )

    # Whether to add LoRA to these additional layers
    add_to_mlp: bool = False
    mlp_modules: Tuple[str, ...] = ('fc1', 'fc2')

    # Bias handling
    bias: str = 'none'  # 'none', 'all', 'lora_only'

    # Initialization
    init_lora_weights: bool = True  # True = Kaiming, False = zeros

    # Merge settings
    merge_weights_on_save: bool = False

    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.rank

    @property
    def all_target_modules(self) -> List[str]:
        """Get all target modules including optional MLP layers."""
        modules = list(self.target_modules)
        if self.add_to_mlp:
            modules.extend(self.mlp_modules)
        return modules

    def estimate_trainable_params(self, model_dim: int = 256) -> Dict:
        """
        Estimate trainable parameters for LoRA.

        Args:
            model_dim: Model embedding dimension

        Returns:
            Dictionary with parameter counts
        """
        # Each LoRA layer adds: rank × (in_dim + out_dim) parameters
        params_per_layer = 2 * self.rank * model_dim
        n_attention_layers = len(self.target_modules)
        n_transformer_layers = 6  # Default SST layers

        attention_params = params_per_layer * n_attention_layers * n_transformer_layers

        mlp_params = 0
        if self.add_to_mlp:
            # MLP is typically 4× hidden dim
            mlp_params = (
                self.rank * (model_dim + 4 * model_dim) +  # fc1
                self.rank * (4 * model_dim + model_dim)    # fc2
            ) * n_transformer_layers

        total = attention_params + mlp_params

        return {
            'attention_params': attention_params,
            'mlp_params': mlp_params,
            'total_lora_params': total,
            'typical_full_model_params': 4_200_000,  # Approximate SST size
            'reduction_percent': (1 - total / 4_200_000) * 100,
        }

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FinetuneConfig:
    """
    Configuration for LoRA fine-tuning on field-specific data.

    Fine-tuning adapts a pre-trained SST model to a specific field
    using a small dataset (100-500 samples) with LoRA.
    """
    # ===== Paths =====
    base_model_path: str = 'checkpoints/pretrain/best_model.pt'
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    output_dir: str = 'adapters/'

    # ===== LoRA Configuration =====
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # ===== Target Nutrients =====
    # Nutrients to predict (personal research targets)
    target_nutrients: Tuple[str, ...] = (
        'K', 'Mg', 'Ca', 'P', 'S', 'Zn', 'Fe', 'Mn', 'Cu', 'B'
    )

    # ===== Training Hyperparameters =====
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Epochs and early stopping
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # LR scheduler
    lr_scheduler: str = 'reduce_on_plateau'  # 'cosine', 'reduce_on_plateau'
    lr_patience: int = 5
    lr_factor: float = 0.5

    # ===== Regularization =====
    gradient_clip_norm: float = 1.0
    dropout_override: Optional[float] = None  # Override model dropout

    # ===== Infrastructure =====
    device: str = 'cuda'
    mixed_precision: bool = True
    num_workers: int = 2

    # ===== Validation =====
    val_every_n_epochs: int = 1
    val_metric: str = 'val_loss'  # Metric for early stopping

    # ===== Field Identification =====
    field_id: Optional[str] = None
    farm_id: Optional[str] = None

    # ===== Logging =====
    use_wandb: bool = True
    wandb_project: str = 'soil-state-transformer'
    wandb_tags: List[str] = field(default_factory=lambda: ['finetune', 'lora'])

    def __post_init__(self):
        """Validate and setup."""
        if self.field_id:
            self.output_dir = f'adapters/{self.field_id}/'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['lora'] = self.lora.to_dict()
        return result


@dataclass
class InferenceConfig:
    """
    Configuration for model inference.
    """
    # ===== Model Paths =====
    base_model_path: str = 'checkpoints/pretrain/best_model.pt'
    adapter_path: Optional[str] = None  # Path to LoRA adapter

    # ===== Inference Settings =====
    batch_size: int = 64
    device: str = 'cuda'

    # Uncertainty estimation
    return_uncertainty: bool = True
    mc_dropout_samples: int = 0  # 0 = no MC dropout, >0 = number of samples

    # ===== Output =====
    output_format: str = 'numpy'  # 'numpy', 'tensor', 'dataframe'

    def to_dict(self) -> Dict:
        return asdict(self)


def load_config(path: Union[str, Path]) -> Union[PretrainConfig, FinetuneConfig]:
    """
    Load configuration from YAML or JSON file.

    Automatically detects config type from contents.

    Args:
        path: Path to config file

    Returns:
        Appropriate config object
    """
    path = Path(path)

    if path.suffix in ('.yaml', '.yml'):
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    # Detect config type
    if 'lora' in data or 'base_model_path' in data:
        if 'lora' in data:
            data['lora'] = LoRAConfig(**data['lora'])
        return FinetuneConfig(**data)
    else:
        if 'model' in data:
            data['model'] = ModelConfig(**data['model'])
        return PretrainConfig(**data)


def save_config(config: Union[PretrainConfig, FinetuneConfig], path: Union[str, Path]):
    """
    Save configuration to file.

    Args:
        config: Configuration object
        path: Output path (extension determines format)
    """
    path = Path(path)
    data = config.to_dict()

    if path.suffix in ('.yaml', '.yml'):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

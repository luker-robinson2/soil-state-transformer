"""
Vertex AI Training Configuration and Job Management

Provides utilities for submitting SST pre-training jobs to Vertex AI
with GPU or TPU acceleration.

Features:
    - GPU training (T4, V100, A100)
    - TPU training (v3, v4)
    - Spot/preemptible instances for cost savings
    - WandB integration
    - GCS checkpoint organization by WandB run ID

Usage:
    >>> from training.vertex_training import VertexTrainingConfig, submit_training_job
    >>> config = VertexTrainingConfig(gpu_type='v100', use_spot=True)
    >>> job = submit_training_job(config, training_args)
    >>> job.wait()
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import CustomJob, CustomContainerTrainingJob
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GPU configurations
GPU_CONFIGS = {
    't4': {
        'machine_type': 'n1-standard-4',
        'accelerator_type': 'NVIDIA_TESLA_T4',
        'accelerator_count': 1,
    },
    'v100': {
        'machine_type': 'n1-standard-8',
        'accelerator_type': 'NVIDIA_TESLA_V100',
        'accelerator_count': 1,
    },
    'a100': {
        'machine_type': 'a2-highgpu-1g',
        'accelerator_type': 'NVIDIA_TESLA_A100',
        'accelerator_count': 1,
    },
    'a100-2x': {
        'machine_type': 'a2-highgpu-2g',
        'accelerator_type': 'NVIDIA_TESLA_A100',
        'accelerator_count': 2,
    },
}

# TPU configurations
TPU_CONFIGS = {
    'v3-8': {
        'machine_type': 'cloud-tpu',
        'accelerator_type': 'TPU_V3',
        'accelerator_count': 8,
    },
    'v4-8': {
        'machine_type': 'cloud-tpu',
        'accelerator_type': 'TPU_V4_POD',
        'accelerator_count': 8,
    },
}


@dataclass
class VertexTrainingConfig:
    """
    Configuration for Vertex AI training jobs.

    Attributes:
        project_id: GCP project ID
        region: GCP region for training
        staging_bucket: GCS bucket for staging artifacts
        gpu_type: GPU type (t4, v100, a100, a100-2x)
        tpu_type: TPU type (v3-8, v4-8) - mutually exclusive with gpu_type
        use_spot: Whether to use preemptible/spot instances
        container_uri: Training container image URI
        service_account: Service account for the training job
    """
    project_id: str = 'your-gcp-project'
    region: str = 'us-central1'
    staging_bucket: str = 'gs://your-research-bucket/vertex-staging'

    # Compute configuration
    gpu_type: Optional[str] = 'v100'
    tpu_type: Optional[str] = None
    use_spot: bool = True

    # Container configuration
    container_uri: str = 'us-central1-docker.pkg.dev/your-gcp-project/sst/sst-training:latest'
    tpu_container_uri: str = 'us-central1-docker.pkg.dev/your-gcp-project/sst/sst-training-tpu:latest'

    # Optional service account
    service_account: Optional[str] = None

    # Timeout (in seconds, default 24 hours)
    timeout: int = 86400

    def __post_init__(self):
        """Validate configuration."""
        if self.gpu_type and self.tpu_type:
            raise ValueError("Cannot specify both gpu_type and tpu_type")

        if self.gpu_type and self.gpu_type not in GPU_CONFIGS:
            raise ValueError(f"Invalid gpu_type: {self.gpu_type}. Options: {list(GPU_CONFIGS.keys())}")

        if self.tpu_type and self.tpu_type not in TPU_CONFIGS:
            raise ValueError(f"Invalid tpu_type: {self.tpu_type}. Options: {list(TPU_CONFIGS.keys())}")

    @property
    def machine_spec(self) -> Dict:
        """Get machine specification for training."""
        if self.tpu_type:
            config = TPU_CONFIGS[self.tpu_type]
        else:
            config = GPU_CONFIGS.get(self.gpu_type, GPU_CONFIGS['v100'])

        return {
            'machine_type': config['machine_type'],
            'accelerator_type': config['accelerator_type'],
            'accelerator_count': config['accelerator_count'],
        }

    @property
    def effective_container_uri(self) -> str:
        """Get the appropriate container URI based on accelerator type."""
        if self.tpu_type:
            return self.tpu_container_uri
        return self.container_uri


def submit_training_job(
    config: VertexTrainingConfig,
    training_args: List[str],
    display_name: str = 'sst-pretrain',
    labels: Optional[Dict[str, str]] = None,
    environment_variables: Optional[Dict[str, str]] = None,
) -> 'CustomJob':
    """
    Submit a training job to Vertex AI.

    Args:
        config: VertexTrainingConfig with job settings
        training_args: Command-line arguments for training script
        display_name: Display name for the job
        labels: Optional labels for the job
        environment_variables: Optional environment variables

    Returns:
        CustomJob object for monitoring
    """
    if not VERTEX_AVAILABLE:
        raise ImportError(
            "google-cloud-aiplatform is required. "
            "Install with: pip install google-cloud-aiplatform"
        )

    # Initialize Vertex AI
    aiplatform.init(
        project=config.project_id,
        location=config.region,
        staging_bucket=config.staging_bucket,
    )

    # Build worker pool spec
    worker_pool_spec = {
        'replica_count': 1,
        'container_spec': {
            'image_uri': config.effective_container_uri,
            'args': training_args,
            'env': [
                {'name': k, 'value': v}
                for k, v in (environment_variables or {}).items()
            ],
        },
    }

    # Add machine spec
    machine_spec = config.machine_spec
    worker_pool_spec['machine_spec'] = {
        'machine_type': machine_spec['machine_type'],
    }

    # Add accelerator if not TPU (TPU uses different config)
    if config.gpu_type:
        worker_pool_spec['machine_spec']['accelerator_type'] = machine_spec['accelerator_type']
        worker_pool_spec['machine_spec']['accelerator_count'] = machine_spec['accelerator_count']

    # Create job
    job = CustomJob(
        display_name=display_name,
        worker_pool_specs=[worker_pool_spec],
        labels=labels or {},
    )

    # Submit job
    logger.info(f"Submitting training job: {display_name}")
    logger.info(f"  Container: {config.effective_container_uri}")
    logger.info(f"  Machine: {machine_spec}")
    logger.info(f"  Spot: {config.use_spot}")
    logger.info(f"  Args: {training_args}")

    job.run(
        service_account=config.service_account,
        timeout=config.timeout,
        restart_job_on_worker_restart=config.use_spot,  # Auto-restart for spot instances
        enable_web_access=True,  # Enable monitoring dashboard
        sync=False,  # Return immediately (don't wait for completion)
    )

    logger.info(f"Job submitted: {job.resource_name}")
    logger.info(f"Dashboard: https://console.cloud.google.com/vertex-ai/training/custom-jobs/{job.resource_name.split('/')[-1]}")

    return job


def get_training_args(
    train_path: str,
    val_path: str,
    gcs_bucket: str = 'your-research-bucket',
    max_steps: int = 50000,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    wandb_project: str = 'soil-state-transformer',
    wandb_run_name: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> List[str]:
    """
    Build command-line arguments for training script.

    Args:
        train_path: GCS path to training data
        val_path: GCS path to validation data
        gcs_bucket: GCS bucket for checkpoints
        max_steps: Maximum training steps
        batch_size: Batch size
        learning_rate: Learning rate
        wandb_project: WandB project name
        wandb_run_name: Optional WandB run name
        resume_from: Optional checkpoint path to resume from

    Returns:
        List of command-line arguments
    """
    args = [
        '--train-path', train_path,
        '--val-path', val_path,
        '--use-gcs',
        '--gcs-bucket', gcs_bucket,
        '--max-steps', str(max_steps),
        '--batch-size', str(batch_size),
        '--lr', str(learning_rate),
    ]

    if wandb_run_name:
        # WandB configuration is handled via environment variables
        pass

    if resume_from:
        args.extend(['--resume', resume_from])

    return args


def get_wandb_env_vars(
    project: str = 'soil-state-transformer',
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Get environment variables for WandB integration.

    Args:
        project: WandB project name
        run_name: Optional run name
        tags: Optional tags for the run

    Returns:
        Dictionary of environment variables
    """
    env = {
        'WANDB_PROJECT': project,
        'WANDB_LOG_MODEL': 'false',  # We handle model saving ourselves
    }

    if run_name:
        env['WANDB_RUN_NAME'] = run_name

    if tags:
        env['WANDB_TAGS'] = ','.join(tags)

    # WANDB_API_KEY should be set via secrets or service account

    return env


def estimate_cost(
    config: VertexTrainingConfig,
    hours: float,
) -> Dict[str, float]:
    """
    Estimate training cost.

    Args:
        config: Training configuration
        hours: Estimated training duration in hours

    Returns:
        Dictionary with cost estimates
    """
    # Approximate hourly costs (subject to change)
    gpu_costs = {
        't4': {'on_demand': 0.35, 'spot': 0.11},
        'v100': {'on_demand': 2.48, 'spot': 0.74},
        'a100': {'on_demand': 3.67, 'spot': 1.10},
        'a100-2x': {'on_demand': 7.34, 'spot': 2.20},
    }

    tpu_costs = {
        'v3-8': {'on_demand': 8.00, 'spot': 2.40},
        'v4-8': {'on_demand': 12.88, 'spot': 3.86},
    }

    if config.tpu_type:
        costs = tpu_costs.get(config.tpu_type, {'on_demand': 0, 'spot': 0})
        accelerator = config.tpu_type
    else:
        costs = gpu_costs.get(config.gpu_type, {'on_demand': 0, 'spot': 0})
        accelerator = config.gpu_type

    rate = costs['spot'] if config.use_spot else costs['on_demand']
    total = rate * hours

    return {
        'accelerator': accelerator,
        'hourly_rate': rate,
        'hours': hours,
        'total_cost': total,
        'pricing_type': 'spot' if config.use_spot else 'on_demand',
    }

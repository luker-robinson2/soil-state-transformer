#!/usr/bin/env python3
"""
Submit SST Pre-training Job to Vertex AI

Submits a training job to Google Cloud Vertex AI with the specified
configuration. Supports GPU (T4, V100, A100) and TPU (v3, v4) training.

Usage:
    # Submit V100 training job with spot pricing
    python scripts/submit_training_job.py --gpu-type v100 --use-spot

    # Submit A100 training job
    python scripts/submit_training_job.py --gpu-type a100

    # Submit TPU training job
    python scripts/submit_training_job.py --tpu v3-8

    # Resume from checkpoint
    python scripts/submit_training_job.py --resume gs://bucket/checkpoints/checkpoint_1000.pt

Example:
    python scripts/submit_training_job.py \\
        --train-path gs://your-research-bucket/data/processed/wosis_train.parquet \\
        --val-path gs://your-research-bucket/data/processed/wosis_val.parquet \\
        --gpu-type v100 \\
        --use-spot \\
        --max-steps 50000 \\
        --wandb-project soil-state-transformer
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.vertex_training import (
    VertexTrainingConfig,
    submit_training_job,
    get_training_args,
    get_wandb_env_vars,
    estimate_cost,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Submit SST pre-training job to Vertex AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data paths
    parser.add_argument(
        '--train-path',
        type=str,
        default='gs://your-research-bucket/data/processed/wosis_train.parquet',
        help='GCS path to training data',
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default='gs://your-research-bucket/data/processed/wosis_val.parquet',
        help='GCS path to validation data',
    )

    # Compute configuration
    parser.add_argument(
        '--gpu-type',
        type=str,
        choices=['t4', 'v100', 'a100', 'a100-2x'],
        default='v100',
        help='GPU type (default: v100)',
    )
    parser.add_argument(
        '--tpu',
        type=str,
        choices=['v3-8', 'v4-8'],
        help='TPU type (mutually exclusive with --gpu-type)',
    )
    parser.add_argument(
        '--use-spot',
        action='store_true',
        default=True,
        help='Use spot/preemptible instances (default: True)',
    )
    parser.add_argument(
        '--no-spot',
        action='store_true',
        help='Disable spot instances (use on-demand)',
    )

    # Training configuration
    parser.add_argument(
        '--max-steps',
        type=int,
        default=50000,
        help='Maximum training steps (default: 50000)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)',
    )

    # Checkpoint configuration
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default='your-research-bucket',
        help='GCS bucket for checkpoints (default: your-research-bucket)',
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='GCS path to checkpoint to resume from',
    )

    # WandB configuration
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='soil-state-transformer',
        help='WandB project name (default: soil-state-transformer)',
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        help='WandB run name (auto-generated if not specified)',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging',
    )

    # Job configuration
    parser.add_argument(
        '--job-name',
        type=str,
        help='Display name for the job (auto-generated if not specified)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration without submitting job',
    )

    args = parser.parse_args()

    # Handle spot instances
    use_spot = args.use_spot and not args.no_spot

    # Create configuration
    config = VertexTrainingConfig(
        gpu_type=None if args.tpu else args.gpu_type,
        tpu_type=args.tpu,
        use_spot=use_spot,
        staging_bucket=f'gs://{args.gcs_bucket}/vertex-staging',
    )

    # Build training arguments
    training_args = get_training_args(
        train_path=args.train_path,
        val_path=args.val_path,
        gcs_bucket=args.gcs_bucket,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume_from=args.resume,
    )

    # Add no-wandb flag if requested
    if args.no_wandb:
        training_args.append('--no-wandb')

    # Build environment variables
    env_vars = {}
    if not args.no_wandb:
        env_vars.update(get_wandb_env_vars(
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            tags=['vertex-ai', args.gpu_type or args.tpu],
        ))

    # Generate job name if not specified
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    accelerator = args.tpu or args.gpu_type
    job_name = args.job_name or f'sst-pretrain-{accelerator}-{timestamp}'

    # Estimate cost
    estimated_hours = args.max_steps * 1.5 / 3600  # ~1.5s per step
    cost_estimate = estimate_cost(config, estimated_hours)

    # Print configuration
    print("\n" + "=" * 60)
    print("SST Pre-training Job Configuration")
    print("=" * 60)
    print(f"Job Name:      {job_name}")
    print(f"Accelerator:   {accelerator}")
    print(f"Spot:          {use_spot}")
    print(f"Max Steps:     {args.max_steps:,}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Train Data:    {args.train_path}")
    print(f"Val Data:      {args.val_path}")
    print(f"Checkpoint:    gs://{args.gcs_bucket}/checkpoints/pretrain/")
    if args.resume:
        print(f"Resume From:   {args.resume}")
    print("")
    print("Estimated Cost:")
    print(f"  Duration:    ~{cost_estimate['hours']:.1f} hours")
    print(f"  Rate:        ${cost_estimate['hourly_rate']:.2f}/hr ({cost_estimate['pricing_type']})")
    print(f"  Total:       ~${cost_estimate['total_cost']:.2f}")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN] Would submit job with args:")
        print(f"  {' '.join(training_args)}")
        return

    # Confirm submission
    print("\nSubmit this job? [y/N] ", end='')
    response = input().strip().lower()
    if response != 'y':
        print("Job submission cancelled.")
        return

    # Submit job
    try:
        job = submit_training_job(
            config=config,
            training_args=training_args,
            display_name=job_name,
            labels={
                'model': 'sst',
                'task': 'pretrain',
                'accelerator': accelerator,
            },
            environment_variables=env_vars,
        )

        print("\n" + "=" * 60)
        print("Job Submitted Successfully!")
        print("=" * 60)
        print(f"Job Name:   {job_name}")
        print(f"Job ID:     {job.resource_name.split('/')[-1]}")
        print(f"Dashboard:  https://console.cloud.google.com/vertex-ai/training/custom-jobs")
        print("")
        print("Monitor with:")
        print(f"  python scripts/monitor_training_job.py --job-id {job.resource_name.split('/')[-1]}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

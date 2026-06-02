#!/usr/bin/env python3
"""
Download Trained Model from GCS

Downloads trained model checkpoints from GCS, organized by WandB run ID.

Usage:
    # Download best model for a specific run
    python scripts/download_model.py --wandb-run-id abc123def

    # Download specific checkpoint
    python scripts/download_model.py --checkpoint gs://bucket/path/checkpoint_50000.pt

    # List available checkpoints
    python scripts/download_model.py --wandb-run-id abc123def --list
"""

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_checkpoints(bucket_name: str, prefix: str):
    """List available checkpoints in GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    print("\n" + "=" * 60)
    print(f"Checkpoints in gs://{bucket_name}/{prefix}")
    print("=" * 60)

    checkpoints = []
    config = None
    stats = None

    for blob in blobs:
        filename = blob.name.split('/')[-1]
        size_mb = blob.size / (1024 * 1024)

        if filename.endswith('.pt'):
            checkpoints.append({
                'name': filename,
                'path': f"gs://{bucket_name}/{blob.name}",
                'size': size_mb,
                'updated': blob.updated,
            })
        elif filename == 'config.json':
            config = f"gs://{bucket_name}/{blob.name}"
        elif filename == 'normalization_stats.json':
            stats = f"gs://{bucket_name}/{blob.name}"

    if not checkpoints:
        print("No checkpoints found.")
        return

    # Sort by step number
    def extract_step(ckpt):
        name = ckpt['name']
        if 'checkpoint_' in name:
            try:
                return int(name.replace('checkpoint_', '').replace('.pt', ''))
            except ValueError:
                return 0
        return -1

    checkpoints.sort(key=extract_step)

    print(f"{'Checkpoint':<30} {'Size (MB)':<12} {'Updated'}")
    print("-" * 60)
    for ckpt in checkpoints:
        updated = ckpt['updated'].strftime('%Y-%m-%d %H:%M') if ckpt['updated'] else 'N/A'
        print(f"{ckpt['name']:<30} {ckpt['size']:<12.1f} {updated}")

    print("-" * 60)
    if config:
        print(f"Config: {config}")
    if stats:
        print(f"Stats:  {stats}")
    print("=" * 60)


def download_file(gcs_uri: str, local_path: Path):
    """Download a file from GCS."""
    client = storage.Client()

    # Parse GCS URI
    if not gcs_uri.startswith('gs://'):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    path = gcs_uri[5:]  # Remove 'gs://'
    bucket_name, blob_path = path.split('/', 1)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Create parent directory
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download
    logger.info(f"Downloading {gcs_uri}...")
    blob.download_to_filename(str(local_path))
    logger.info(f"  -> {local_path} ({local_path.stat().st_size / 1e6:.1f} MB)")

    return local_path


def download_model(
    bucket_name: str,
    prefix: str,
    local_dir: Path,
    download_best: bool = True,
    download_latest: bool = False,
    specific_checkpoint: str = None,
):
    """Download model files from GCS."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Find checkpoints
    blobs = list(bucket.list_blobs(prefix=prefix))

    if not blobs:
        logger.error(f"No files found at gs://{bucket_name}/{prefix}")
        return

    downloaded = []

    # Download config and stats first
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        if filename in ['config.json', 'normalization_stats.json']:
            local_path = local_dir / filename
            download_file(f"gs://{bucket_name}/{blob.name}", local_path)
            downloaded.append(local_path)

    # Find and download checkpoint(s)
    checkpoints = [b for b in blobs if b.name.endswith('.pt')]

    if specific_checkpoint:
        # Download specific checkpoint
        for blob in checkpoints:
            if specific_checkpoint in blob.name:
                local_path = local_dir / blob.name.split('/')[-1]
                download_file(f"gs://{bucket_name}/{blob.name}", local_path)
                downloaded.append(local_path)
                break
        else:
            logger.warning(f"Checkpoint not found: {specific_checkpoint}")

    elif download_best:
        # Download best_model.pt if exists
        best_blob = next((b for b in checkpoints if 'best_model.pt' in b.name), None)
        if best_blob:
            local_path = local_dir / 'best_model.pt'
            download_file(f"gs://{bucket_name}/{best_blob.name}", local_path)
            downloaded.append(local_path)
        else:
            logger.warning("best_model.pt not found")

    if download_latest:
        # Download latest checkpoint
        step_checkpoints = [b for b in checkpoints if 'checkpoint_' in b.name]
        if step_checkpoints:
            # Sort by step number
            def extract_step(blob):
                name = blob.name.split('/')[-1]
                try:
                    return int(name.replace('checkpoint_', '').replace('.pt', ''))
                except ValueError:
                    return 0

            step_checkpoints.sort(key=extract_step, reverse=True)
            latest = step_checkpoints[0]
            local_path = local_dir / latest.name.split('/')[-1]
            download_file(f"gs://{bucket_name}/{latest.name}", local_path)
            downloaded.append(local_path)

    return downloaded


def main():
    parser = argparse.ArgumentParser(description='Download trained model from GCS')

    parser.add_argument(
        '--wandb-run-id',
        type=str,
        help='WandB run ID to download',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Specific GCS checkpoint path to download',
    )
    parser.add_argument(
        '--local-path',
        type=str,
        default='./checkpoints/downloaded/',
        help='Local directory to save model (default: ./checkpoints/downloaded/)',
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default='your-research-bucket',
        help='GCS bucket name (default: your-research-bucket)',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='checkpoints/pretrain/',
        help='Checkpoint prefix (default: checkpoints/pretrain/)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available checkpoints',
    )
    parser.add_argument(
        '--best',
        action='store_true',
        default=True,
        help='Download best_model.pt (default)',
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Also download latest checkpoint',
    )

    args = parser.parse_args()

    if not GCS_AVAILABLE:
        logger.error("google-cloud-storage is required. Install with: pip install google-cloud-storage")
        sys.exit(1)

    # Construct prefix
    if args.wandb_run_id:
        prefix = f"{args.prefix.rstrip('/')}/{args.wandb_run_id}/"
    else:
        prefix = args.prefix

    if args.list:
        list_checkpoints(args.bucket, prefix)
        return

    if args.checkpoint:
        # Download specific checkpoint
        local_path = Path(args.local_path) / args.checkpoint.split('/')[-1]
        download_file(args.checkpoint, local_path)
        print(f"\nDownloaded: {local_path}")
        return

    if not args.wandb_run_id:
        parser.error("--wandb-run-id is required (or use --list to see runs)")

    # Download model
    local_dir = Path(args.local_path) / args.wandb_run_id
    downloaded = download_model(
        bucket_name=args.bucket,
        prefix=prefix,
        local_dir=local_dir,
        download_best=args.best,
        download_latest=args.latest,
    )

    if downloaded:
        print("\n" + "=" * 60)
        print("Downloaded Files")
        print("=" * 60)
        for path in downloaded:
            print(f"  {path}")
        print("=" * 60)

        print("\nUsage example:")
        print(f"""
from models.sst_pretrain import SSTPretrainModel
import torch

# Load checkpoint
checkpoint = torch.load('{local_dir}/best_model.pt')
model = SSTPretrainModel(n_properties=8)
model.load_state_dict(checkpoint['model_state_dict'])

# Get normalization stats
import json
with open('{local_dir}/normalization_stats.json') as f:
    norm_stats = json.load(f)
""")


if __name__ == '__main__':
    main()

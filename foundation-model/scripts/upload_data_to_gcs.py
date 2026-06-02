#!/usr/bin/env python3
"""
Upload Training Data to GCS

Uploads local WoSIS training data to Google Cloud Storage for cloud training.

Usage:
    python scripts/upload_data_to_gcs.py
    python scripts/upload_data_to_gcs.py --bucket my-bucket --prefix data/

Files uploaded:
    - data/processed/wosis_train.parquet
    - data/processed/wosis_val.parquet
    - data/processed/wosis_test.parquet (optional)
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_file_to_gcs(
    local_path: Path,
    bucket: storage.Bucket,
    blob_path: str,
) -> str:
    """
    Upload a single file to GCS.

    Args:
        local_path: Local file path
        bucket: GCS bucket object
        blob_path: Destination path within bucket

    Returns:
        GCS URI of uploaded file
    """
    blob = bucket.blob(blob_path)

    # Upload with progress
    file_size = local_path.stat().st_size
    logger.info(f"Uploading {local_path.name} ({file_size / 1e6:.1f} MB)...")

    blob.upload_from_filename(str(local_path))

    gcs_uri = f"gs://{bucket.name}/{blob_path}"
    logger.info(f"  -> {gcs_uri}")

    return gcs_uri


def upload_training_data(
    bucket_name: str = 'your-research-bucket',
    prefix: str = 'data/processed/',
    data_dir: str = 'data/processed/',
):
    """
    Upload all training data to GCS.

    Args:
        bucket_name: GCS bucket name
        prefix: Destination prefix within bucket
        data_dir: Local directory containing training data
    """
    if not GCS_AVAILABLE:
        logger.error("google-cloud-storage not available. Install with: pip install google-cloud-storage")
        sys.exit(1)

    # Initialize client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Ensure bucket exists
    if not bucket.exists():
        logger.error(f"Bucket {bucket_name} does not exist")
        sys.exit(1)

    # Find data directory
    script_dir = Path(__file__).parent.parent
    data_path = script_dir / data_dir

    if not data_path.exists():
        logger.error(f"Data directory not found: {data_path}")
        sys.exit(1)

    # Files to upload
    files = [
        'wosis_train.parquet',
        'wosis_val.parquet',
        'wosis_test.parquet',
    ]

    uploaded = []
    for filename in files:
        local_file = data_path / filename
        if local_file.exists():
            blob_path = f"{prefix.rstrip('/')}/{filename}"
            gcs_uri = upload_file_to_gcs(local_file, bucket, blob_path)
            uploaded.append(gcs_uri)
        else:
            logger.warning(f"File not found, skipping: {local_file}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"Uploaded {len(uploaded)} files to gs://{bucket_name}/{prefix}")
    for uri in uploaded:
        logger.info(f"  {uri}")
    logger.info("=" * 60)

    # Print usage instructions
    print("\n" + "=" * 60)
    print("To use this data for cloud training:")
    print("=" * 60)
    print(f"""
python -m training.pretrain \\
    --train-path gs://{bucket_name}/{prefix}wosis_train.parquet \\
    --val-path gs://{bucket_name}/{prefix}wosis_val.parquet \\
    --use-gcs \\
    --gcs-bucket {bucket_name}
""")


def main():
    parser = argparse.ArgumentParser(
        description='Upload WoSIS training data to GCS'
    )
    parser.add_argument(
        '--bucket',
        type=str,
        default='your-research-bucket',
        help='GCS bucket name (default: your-research-bucket)'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='data/processed/',
        help='Destination prefix within bucket (default: data/processed/)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed/',
        help='Local data directory (default: data/processed/)'
    )

    args = parser.parse_args()

    upload_training_data(
        bucket_name=args.bucket,
        prefix=args.prefix,
        data_dir=args.data_dir,
    )


if __name__ == '__main__':
    main()

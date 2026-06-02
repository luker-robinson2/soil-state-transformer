"""
GCS Checkpoint Manager for Cloud Training

Provides transparent checkpoint save/load to Google Cloud Storage,
organized by WandB run ID for easy experiment tracking.

Features:
    - Automatic retry with exponential backoff
    - Streaming upload for large checkpoints
    - Support for both local and GCS paths
    - Integration with WandB run IDs

Usage:
    >>> from training.gcs_storage import GCSCheckpointManager
    >>> manager = GCSCheckpointManager(
    ...     bucket='your-research-bucket',
    ...     wandb_run_id='abc123def',
    ... )
    >>> gcs_path = manager.save_checkpoint(checkpoint, step=1000)
    >>> checkpoint = manager.load_checkpoint(gcs_path)
"""

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# Optional GCS imports
try:
    from google.cloud import storage
    from google.api_core import retry
    from google.oauth2 import service_account
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

logger = logging.getLogger(__name__)


class GCSCheckpointManager:
    """
    Manages checkpoint save/load to GCS, organized by WandB run ID.

    Checkpoint paths follow the format:
        gs://{bucket}/checkpoints/pretrain/{wandb_run_id}/checkpoint_{step}.pt

    Attributes:
        bucket_name: GCS bucket name
        wandb_run_id: WandB run ID for organizing checkpoints
        prefix: Base path prefix within bucket
        client: GCS storage client
    """

    def __init__(
        self,
        bucket: str,
        wandb_run_id: Optional[str] = None,
        prefix: str = 'checkpoints/pretrain/',
        credentials_path: Optional[str] = None,
    ):
        """
        Initialize GCS checkpoint manager.

        Args:
            bucket: GCS bucket name (without gs:// prefix)
            wandb_run_id: WandB run ID for checkpoint organization
            prefix: Base path prefix within bucket
            credentials_path: Path to service account JSON (optional)
        """
        if not GCS_AVAILABLE:
            raise ImportError(
                "google-cloud-storage is required for GCS operations. "
                "Install with: pip install google-cloud-storage"
            )

        self.bucket_name = bucket
        self.wandb_run_id = wandb_run_id or 'local'
        self.prefix = prefix.rstrip('/')

        # Initialize GCS client
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = storage.Client(credentials=credentials)
        else:
            # Use default credentials (ADC or GOOGLE_APPLICATION_CREDENTIALS)
            self.client = storage.Client()

        self.bucket = self.client.bucket(bucket)

        logger.info(
            f"GCS Checkpoint Manager initialized: "
            f"gs://{bucket}/{self.prefix}/{self.wandb_run_id}/"
        )

    @property
    def checkpoint_prefix(self) -> str:
        """Get the full prefix for checkpoints including run ID."""
        return f"{self.prefix}/{self.wandb_run_id}"

    def _get_blob_path(self, filename: str) -> str:
        """Get full blob path for a checkpoint file."""
        return f"{self.checkpoint_prefix}/{filename}"

    @retry.Retry(predicate=retry.if_exception_type(Exception), deadline=300)
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        step: int,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint to GCS.

        Args:
            checkpoint: Checkpoint dictionary to save
            step: Training step number
            is_best: Whether this is the best model so far

        Returns:
            GCS URI of saved checkpoint
        """
        # Save step checkpoint
        filename = f"checkpoint_{step}.pt"
        gcs_path = self._get_blob_path(filename)

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        buffer.seek(0)

        # Upload with retry
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Saved checkpoint to {gcs_uri}")

        # Also save as best model if applicable
        if is_best:
            best_path = self._get_blob_path("best_model.pt")
            buffer.seek(0)
            best_blob = self.bucket.blob(best_path)
            best_blob.upload_from_file(buffer, content_type='application/octet-stream')
            logger.info(f"Saved best model to gs://{self.bucket_name}/{best_path}")

        return gcs_uri

    @retry.Retry(predicate=retry.if_exception_type(Exception), deadline=300)
    def load_checkpoint(
        self,
        checkpoint_path: str,
        map_location: str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Load checkpoint from GCS or local path.

        Args:
            checkpoint_path: GCS URI (gs://...) or local path
            map_location: Device to map tensors to

        Returns:
            Loaded checkpoint dictionary
        """
        if checkpoint_path.startswith('gs://'):
            # Parse GCS URI
            path = checkpoint_path.replace(f'gs://{self.bucket_name}/', '')

            # Download to temp file
            blob = self.bucket.blob(path)

            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                blob.download_to_filename(tmp.name)
                checkpoint = torch.load(tmp.name, map_location=map_location)
                os.unlink(tmp.name)

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
        else:
            # Local path
            return torch.load(checkpoint_path, map_location=map_location)

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.

        Returns:
            GCS URI of latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        # Sort by step number (extract from checkpoint_{step}.pt)
        def extract_step(path: str) -> int:
            filename = path.split('/')[-1]
            if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                try:
                    return int(filename[11:-3])
                except ValueError:
                    return -1
            return -1

        checkpoints.sort(key=extract_step, reverse=True)
        return checkpoints[0] if checkpoints else None

    def list_checkpoints(self) -> List[str]:
        """
        List all checkpoints in GCS.

        Returns:
            List of GCS URIs for all checkpoints
        """
        blobs = self.bucket.list_blobs(prefix=self.checkpoint_prefix)

        checkpoints = []
        for blob in blobs:
            if blob.name.endswith('.pt') and 'checkpoint_' in blob.name:
                checkpoints.append(f"gs://{self.bucket_name}/{blob.name}")

        return checkpoints

    def get_best_model_path(self) -> Optional[str]:
        """
        Get the path to the best model checkpoint.

        Returns:
            GCS URI of best model, or None if not found
        """
        best_path = self._get_blob_path("best_model.pt")
        blob = self.bucket.blob(best_path)

        if blob.exists():
            return f"gs://{self.bucket_name}/{best_path}"
        return None

    def save_config(self, config: Dict[str, Any]) -> str:
        """
        Save training configuration to GCS.

        Args:
            config: Configuration dictionary

        Returns:
            GCS URI of saved config
        """
        config_path = self._get_blob_path("config.json")
        blob = self.bucket.blob(config_path)
        blob.upload_from_string(
            json.dumps(config, indent=2),
            content_type='application/json'
        )

        gcs_uri = f"gs://{self.bucket_name}/{config_path}"
        logger.info(f"Saved config to {gcs_uri}")
        return gcs_uri

    def save_normalization_stats(self, stats: Dict[str, Any]) -> str:
        """
        Save normalization statistics to GCS.

        Args:
            stats: Normalization statistics dictionary

        Returns:
            GCS URI of saved stats
        """
        stats_path = self._get_blob_path("normalization_stats.json")
        blob = self.bucket.blob(stats_path)
        blob.upload_from_string(
            json.dumps(stats, indent=2),
            content_type='application/json'
        )

        gcs_uri = f"gs://{self.bucket_name}/{stats_path}"
        logger.info(f"Saved normalization stats to {gcs_uri}")
        return gcs_uri

    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> int:
        """
        Remove old checkpoints, keeping only the last N.

        Args:
            keep_last_n: Number of recent checkpoints to keep

        Returns:
            Number of checkpoints deleted
        """
        checkpoints = self.list_checkpoints()

        # Sort by step number
        def extract_step(path: str) -> int:
            filename = path.split('/')[-1]
            if filename.startswith('checkpoint_') and filename.endswith('.pt'):
                try:
                    return int(filename[11:-3])
                except ValueError:
                    return -1
            return -1

        checkpoints.sort(key=extract_step)

        # Delete all but the last N
        to_delete = checkpoints[:-keep_last_n] if len(checkpoints) > keep_last_n else []
        deleted = 0

        for ckpt_uri in to_delete:
            path = ckpt_uri.replace(f'gs://{self.bucket_name}/', '')
            blob = self.bucket.blob(path)
            blob.delete()
            deleted += 1
            logger.debug(f"Deleted old checkpoint: {ckpt_uri}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old checkpoints")

        return deleted

    def download_to_local(
        self,
        gcs_path: str,
        local_path: Union[str, Path],
    ) -> Path:
        """
        Download a file from GCS to local path.

        Args:
            gcs_path: GCS URI or blob path
            local_path: Local destination path

        Returns:
            Path to downloaded file
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if gcs_path.startswith('gs://'):
            path = gcs_path.replace(f'gs://{self.bucket_name}/', '')
        else:
            path = gcs_path

        blob = self.bucket.blob(path)
        blob.download_to_filename(str(local_path))

        logger.info(f"Downloaded {gcs_path} to {local_path}")
        return local_path


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS URI."""
    return isinstance(path, str) and path.startswith('gs://')


def parse_gcs_uri(uri: str) -> tuple:
    """
    Parse a GCS URI into bucket and path.

    Args:
        uri: GCS URI (gs://bucket/path/to/file)

    Returns:
        Tuple of (bucket_name, blob_path)
    """
    if not uri.startswith('gs://'):
        raise ValueError(f"Invalid GCS URI: {uri}")

    path = uri[5:]  # Remove 'gs://'
    parts = path.split('/', 1)

    if len(parts) == 1:
        return parts[0], ''
    return parts[0], parts[1]


# Convenience function for loading checkpoints from either local or GCS
def load_checkpoint_auto(
    path: str,
    bucket: Optional[str] = None,
    map_location: str = 'cpu',
) -> Dict[str, Any]:
    """
    Load checkpoint from local path or GCS URI.

    Args:
        path: Local path or GCS URI
        bucket: GCS bucket (only needed if path is relative)
        map_location: Device to map tensors to

    Returns:
        Loaded checkpoint dictionary
    """
    if is_gcs_path(path):
        bucket_name, blob_path = parse_gcs_uri(path)
        manager = GCSCheckpointManager(bucket=bucket_name)
        return manager.load_checkpoint(path, map_location=map_location)
    else:
        return torch.load(path, map_location=map_location)

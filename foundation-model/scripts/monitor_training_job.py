#!/usr/bin/env python3
"""
Monitor Vertex AI Training Job

Monitors the progress of a Vertex AI training job, streaming logs
and displaying training metrics.

Usage:
    python scripts/monitor_training_job.py --job-id <job-id>
    python scripts/monitor_training_job.py --list  # List recent jobs
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def list_recent_jobs(project_id: str, region: str, limit: int = 10):
    """List recent training jobs."""
    aiplatform.init(project=project_id, location=region)

    jobs = aiplatform.CustomJob.list(
        order_by="create_time desc",
        filter='display_name:"sst-"',  # Filter to SST jobs
    )

    print("\n" + "=" * 80)
    print("Recent SST Training Jobs")
    print("=" * 80)
    print(f"{'Job ID':<30} {'Status':<15} {'Created':<20} {'Display Name'}")
    print("-" * 80)

    for i, job in enumerate(jobs[:limit]):
        job_id = job.resource_name.split('/')[-1]
        status = job.state.name if hasattr(job.state, 'name') else str(job.state)
        created = job.create_time.strftime('%Y-%m-%d %H:%M') if job.create_time else 'N/A'
        print(f"{job_id:<30} {status:<15} {created:<20} {job.display_name}")

    print("=" * 80)


def get_job(project_id: str, region: str, job_id: str):
    """Get a specific job by ID."""
    aiplatform.init(project=project_id, location=region)

    # Construct full resource name if just ID provided
    if not job_id.startswith('projects/'):
        resource_name = f"projects/{project_id}/locations/{region}/customJobs/{job_id}"
    else:
        resource_name = job_id

    return aiplatform.CustomJob.get(resource_name=resource_name)


def monitor_job(job, stream_logs: bool = True):
    """Monitor a training job."""
    print("\n" + "=" * 60)
    print("Training Job Monitor")
    print("=" * 60)
    print(f"Job Name:   {job.display_name}")
    print(f"Job ID:     {job.resource_name.split('/')[-1]}")
    print(f"Status:     {job.state.name if hasattr(job.state, 'name') else job.state}")
    print(f"Created:    {job.create_time}")

    if job.start_time:
        print(f"Started:    {job.start_time}")
        elapsed = datetime.now(job.start_time.tzinfo) - job.start_time
        print(f"Elapsed:    {elapsed}")

    print("=" * 60)

    # Check if job is still running
    running_states = ['JOB_STATE_PENDING', 'JOB_STATE_RUNNING', 'JOB_STATE_QUEUED']
    state_name = job.state.name if hasattr(job.state, 'name') else str(job.state)

    if state_name not in running_states:
        print(f"\nJob has ended with state: {state_name}")
        if job.error:
            print(f"Error: {job.error}")
        return

    if not stream_logs:
        print("\nJob is running. Use --stream to watch logs.")
        return

    # Stream logs (simplified - actual log streaming requires Cloud Logging API)
    print("\nStreaming job status (press Ctrl+C to stop)...")
    print("-" * 60)

    try:
        last_state = state_name
        while True:
            # Refresh job state
            job = get_job(
                project_id=job.resource_name.split('/')[1],
                region=job.resource_name.split('/')[3],
                job_id=job.resource_name.split('/')[-1],
            )

            current_state = job.state.name if hasattr(job.state, 'name') else str(job.state)

            if current_state != last_state:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] State changed: {last_state} -> {current_state}")
                last_state = current_state

            if current_state not in running_states:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Job completed with state: {current_state}")
                if job.error:
                    print(f"Error: {job.error}")
                break

            # Print elapsed time
            if job.start_time:
                elapsed = datetime.now(job.start_time.tzinfo) - job.start_time
                hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Running... Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}  ", end='', flush=True)

            time.sleep(30)  # Poll every 30 seconds

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def cancel_job(job):
    """Cancel a running job."""
    print(f"\nCancelling job: {job.display_name}")
    print("Are you sure? [y/N] ", end='')
    response = input().strip().lower()

    if response == 'y':
        job.cancel()
        print("Job cancellation requested.")
    else:
        print("Cancelled.")


def main():
    parser = argparse.ArgumentParser(description='Monitor Vertex AI training job')

    parser.add_argument(
        '--job-id',
        type=str,
        help='Job ID to monitor',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List recent jobs',
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        default=True,
        help='Stream job status updates',
    )
    parser.add_argument(
        '--cancel',
        action='store_true',
        help='Cancel the job',
    )
    parser.add_argument(
        '--project',
        type=str,
        default='your-gcp-project',
        help='GCP project ID',
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-central1',
        help='GCP region',
    )

    args = parser.parse_args()

    if not VERTEX_AVAILABLE:
        logger.error("google-cloud-aiplatform is required. Install with: pip install google-cloud-aiplatform")
        sys.exit(1)

    if args.list:
        list_recent_jobs(args.project, args.region)
        return

    if not args.job_id:
        parser.error("--job-id is required (or use --list to see jobs)")

    try:
        job = get_job(args.project, args.region, args.job_id)

        if args.cancel:
            cancel_job(job)
        else:
            monitor_job(job, stream_logs=args.stream)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/bin/bash
# Build and push SST training containers to Google Artifact Registry
#
# Usage:
#   ./scripts/build_training_container.sh          # Build GPU container
#   ./scripts/build_training_container.sh --tpu    # Build TPU container
#   ./scripts/build_training_container.sh --all    # Build both containers
#
# Prerequisites:
#   - Docker installed
#   - gcloud CLI authenticated
#   - Artifact Registry API enabled

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-gcp-project}"
REGION="${REGION:-us-central1}"
REGISTRY="${REGISTRY:-us-central1-docker.pkg.dev}"
REPOSITORY="${REPOSITORY:-cloud-run-source-deploy}"
IMAGE_NAME="sst-training"
TPU_IMAGE_NAME="sst-training-tpu"

# Build timestamp tag
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Full image paths
GPU_IMAGE="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}"
TPU_IMAGE="${REGISTRY}/${PROJECT_ID}/${REPOSITORY}/${TPU_IMAGE_NAME}"

# Change to the foundation-model-research directory
cd "$(dirname "$0")/.."

echo "=============================================="
echo "SST Training Container Build"
echo "=============================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Registry: ${REGISTRY}"
echo ""

# Parse arguments
BUILD_GPU=false
BUILD_TPU=false

case "${1:-gpu}" in
    --gpu)
        BUILD_GPU=true
        ;;
    --tpu)
        BUILD_TPU=true
        ;;
    --all)
        BUILD_GPU=true
        BUILD_TPU=true
        ;;
    *)
        BUILD_GPU=true
        ;;
esac

# Configure Docker for Artifact Registry
echo "Configuring Docker authentication..."
gcloud auth configure-docker ${REGISTRY} --quiet

# Build GPU container
if [ "$BUILD_GPU" = true ]; then
    echo ""
    echo "Building GPU training container..."
    echo "Image: ${GPU_IMAGE}:latest"
    echo ""

    docker build \
        -f docker/Dockerfile.training \
        -t ${GPU_IMAGE}:latest \
        -t ${GPU_IMAGE}:${TIMESTAMP} \
        .

    echo ""
    echo "Pushing GPU container to Artifact Registry..."
    docker push ${GPU_IMAGE}:latest
    docker push ${GPU_IMAGE}:${TIMESTAMP}

    echo ""
    echo "GPU container pushed successfully!"
    echo "  ${GPU_IMAGE}:latest"
    echo "  ${GPU_IMAGE}:${TIMESTAMP}"
fi

# Build TPU container
if [ "$BUILD_TPU" = true ]; then
    echo ""
    echo "Building TPU training container..."
    echo "Image: ${TPU_IMAGE}:latest"
    echo ""

    docker build \
        -f docker/Dockerfile.training-tpu \
        -t ${TPU_IMAGE}:latest \
        -t ${TPU_IMAGE}:${TIMESTAMP} \
        .

    echo ""
    echo "Pushing TPU container to Artifact Registry..."
    docker push ${TPU_IMAGE}:latest
    docker push ${TPU_IMAGE}:${TIMESTAMP}

    echo ""
    echo "TPU container pushed successfully!"
    echo "  ${TPU_IMAGE}:latest"
    echo "  ${TPU_IMAGE}:${TIMESTAMP}"
fi

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Upload training data:"
echo "     python scripts/upload_data_to_gcs.py"
echo ""
echo "  2. Submit training job:"
echo "     python scripts/submit_training_job.py --gpu-type v100 --use-spot"
echo ""

#!/bin/bash
# Setup GCS bucket for Foundation Model Research
# Usage: ./setup_gcs.sh

set -e

PROJECT_ID="your-gcp-project"
BUCKET_NAME="your-research-bucket"
REGION="us-central1"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "========================================"
echo "GCS Bucket Setup for Foundation Model"
echo "========================================"

# Check gcloud is configured
echo_info "Checking GCP configuration..."
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo_warn "Current project is '$CURRENT_PROJECT', switching to '$PROJECT_ID'"
    gcloud config set project "$PROJECT_ID"
fi

# Check if bucket exists
if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
    echo_info "Bucket gs://$BUCKET_NAME already exists"
else
    echo_info "Creating bucket gs://$BUCKET_NAME..."
    gcloud storage buckets create "gs://$BUCKET_NAME" \
        --project="$PROJECT_ID" \
        --location="$REGION" \
        --uniform-bucket-level-access
    echo_info "✓ Bucket created"
fi

# Apply lifecycle policies
echo_info "Applying lifecycle policies..."
cat > /tmp/lifecycle.json << 'EOF'
{
  "rule": [
    {
      "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
      "condition": {"age": 30, "matchesPrefix": ["data/raw/"]}
    },
    {
      "action": {"type": "SetStorageClass", "storageClass": "COLDLINE"},
      "condition": {"age": 90, "matchesPrefix": ["experiments/"]}
    },
    {
      "action": {"type": "Delete"},
      "condition": {"age": 7, "matchesPrefix": ["tmp/"]}
    }
  ]
}
EOF

gcloud storage buckets update "gs://$BUCKET_NAME" --lifecycle-file=/tmp/lifecycle.json
echo_info "✓ Lifecycle policies applied"

# Create directory structure
echo_info "Creating directory structure..."
for dir in data/raw/wosis data/raw/era5 data/raw/soilgrids data/processed experiments tmp models; do
    echo "  Creating $dir/"
    echo "" | gsutil cp - "gs://$BUCKET_NAME/$dir/.keep" 2>/dev/null || true
done
echo_info "✓ Directory structure created"

# Verify setup
echo ""
echo "========================================"
echo_info "Setup complete!"
echo "========================================"
echo ""
echo "Bucket: gs://$BUCKET_NAME"
echo "Region: $REGION"
echo ""
echo "Lifecycle policies:"
echo "  - data/raw/* → Nearline after 30 days"
echo "  - experiments/* → Coldline after 90 days"
echo "  - tmp/* → Deleted after 7 days"
echo ""
echo "Next steps:"
echo "  1. Configure ERA5 credentials (see README.md)"
echo "  2. Run ./download_all_data.sh"

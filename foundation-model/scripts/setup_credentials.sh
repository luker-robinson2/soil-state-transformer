#!/bin/bash
# Interactive credential setup for Foundation Model Research
# Usage: ./setup_credentials.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }
echo_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

echo "========================================"
echo "Foundation Model Credentials Setup"
echo "========================================"
echo ""

# ============================================
# GCP Credentials
# ============================================
echo_step "1. Google Cloud Platform (GCP) Credentials"
echo ""

if gcloud auth application-default print-access-token &>/dev/null; then
    echo_info "✓ GCP Application Default Credentials already configured"
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    echo "  Current project: $CURRENT_PROJECT"
else
    echo_warn "GCP credentials not found"
    echo ""
    echo "To configure GCP credentials:"
    echo "  1. Run: gcloud auth application-default login"
    echo "  2. Follow the browser prompts to authenticate"
    echo "  3. Run this script again to verify"
    echo ""
    read -p "Would you like to configure GCP now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud auth application-default login
    fi
fi

echo ""

# ============================================
# ERA5 CDS API Credentials
# ============================================
echo_step "2. ERA5 Climate Data Store (CDS) API Credentials"
echo ""

if [ -f "$HOME/.cdsapirc" ]; then
    echo_info "✓ ERA5 CDS API credentials found at ~/.cdsapirc"
    echo "  Contents:"
    cat "$HOME/.cdsapirc" | sed 's/key:.*/key: [REDACTED]/'
else
    echo_warn "ERA5 credentials not found"
    echo ""
    echo "ERA5 provides global climate reanalysis data essential for"
    echo "temporal features in the foundation model."
    echo ""
    echo "Setup steps:"
    echo "  1. Register at: https://cds.climate.copernicus.eu/"
    echo "  2. Accept license at: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels"
    echo "  3. Get API key from: https://cds.climate.copernicus.eu/profile"
    echo ""
    read -p "Do you have your CDS API credentials? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        read -p "Enter your CDS UID (numeric): " CDS_UID
        read -p "Enter your CDS API key: " CDS_KEY

        cat > "$HOME/.cdsapirc" << EOF
url: https://cds.climate.copernicus.eu/api
key: ${CDS_UID}:${CDS_KEY}
EOF
        chmod 600 "$HOME/.cdsapirc"
        echo_info "✓ ERA5 credentials saved to ~/.cdsapirc"
    else
        echo ""
        echo "You can set up ERA5 credentials later by running this script again"
        echo "or by manually creating ~/.cdsapirc with:"
        echo ""
        echo "  url: https://cds.climate.copernicus.eu/api"
        echo "  key: YOUR_UID:YOUR_API_KEY"
    fi
fi

echo ""

# ============================================
# Summary
# ============================================
echo "========================================"
echo "Credentials Summary"
echo "========================================"
echo ""

# GCP Status
if gcloud auth application-default print-access-token &>/dev/null; then
    echo -e "GCP:  ${GREEN}✓ Configured${NC}"
else
    echo -e "GCP:  ${RED}✗ Not configured${NC}"
fi

# ERA5 Status
if [ -f "$HOME/.cdsapirc" ]; then
    echo -e "ERA5: ${GREEN}✓ Configured${NC}"
else
    echo -e "ERA5: ${YELLOW}○ Optional (not configured)${NC}"
fi

echo ""
echo "Next steps:"
echo "  1. Run: ./download_all_data.sh"
echo "  2. Open notebooks/data_exploration/01_wosis_exploration.ipynb"
echo ""

#!/bin/bash
# Download all data for Foundation Model Research
# Usage: ./download_all_data.sh [--wosis-only] [--era5-only] [--soilgrids-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
VENV_DIR="$PROJECT_DIR/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Python environment
check_environment() {
    echo_info "Checking environment..."

    if [ ! -d "$VENV_DIR" ]; then
        echo_info "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    echo_info "Installing dependencies..."
    pip install -q -r "$PROJECT_DIR/requirements.txt"
}

# Check credentials
check_credentials() {
    echo_info "Checking credentials..."

    # Check GCP credentials
    if ! gcloud auth application-default print-access-token &>/dev/null; then
        echo_error "GCP credentials not configured. Run: gcloud auth application-default login"
        exit 1
    fi
    echo_info "✓ GCP credentials OK"

    # Check ERA5 credentials (optional for WoSIS/SoilGrids)
    if [ -f "$HOME/.cdsapirc" ]; then
        echo_info "✓ ERA5 CDS API credentials found"
    else
        echo_warn "ERA5 credentials not found at ~/.cdsapirc"
        echo_warn "ERA5 download will be skipped. See README.md for setup instructions."
    fi
}

# Download WoSIS data
download_wosis() {
    echo_info "Downloading WoSIS soil profiles..."
    cd "$DATA_DIR"
    python -c "
from wosis_downloader import WoSISDownloader
downloader = WoSISDownloader()

# Download snapshot
print('Downloading WoSIS snapshot...')
raw_path = downloader.download_snapshot()
print(f'Downloaded to: {raw_path}')

# Process data
print('Processing WoSIS data...')
processed_path = downloader.process_snapshot(raw_path)
print(f'Processed to: {processed_path}')

# Upload to GCS
print('Uploading to GCS...')
gcs_path = downloader.upload_to_gcs(processed_path)
print(f'Uploaded to: {gcs_path}')
"
    echo_info "✓ WoSIS download complete"
}

# Download ERA5 data
download_era5() {
    if [ ! -f "$HOME/.cdsapirc" ]; then
        echo_warn "Skipping ERA5 download (no credentials)"
        return
    fi

    echo_info "Downloading ERA5 climate data..."
    cd "$DATA_DIR"
    python -c "
from era5_downloader import ERA5Downloader
downloader = ERA5Downloader()

# Download recent years
for year in [2022, 2023, 2024]:
    print(f'Downloading ERA5 for {year}...')
    try:
        path = downloader.download_for_year(year)
        print(f'  Downloaded: {path}')
    except Exception as e:
        print(f'  Error: {e}')
"
    echo_info "✓ ERA5 download complete"
}

# Download SoilGrids data
download_soilgrids() {
    echo_info "Downloading SoilGrids covariates..."
    cd "$DATA_DIR"
    python -c "
from soilgrids_downloader import SoilGridsDownloader
downloader = SoilGridsDownloader()

# Download key layers
layers = ['soc', 'phh2o', 'clay', 'sand', 'nitrogen']
depths = ['0-5cm', '5-15cm', '15-30cm']

for layer in layers:
    for depth in depths:
        print(f'Downloading {layer} at {depth}...')
        try:
            path = downloader.download_layer(layer, depth)
            print(f'  Downloaded: {path}')
        except Exception as e:
            print(f'  Error: {e}')
"
    echo_info "✓ SoilGrids download complete"
}

# Main execution
main() {
    echo "========================================"
    echo "Foundation Model Data Download"
    echo "========================================"

    check_environment
    check_credentials

    # Parse arguments
    DOWNLOAD_WOSIS=true
    DOWNLOAD_ERA5=true
    DOWNLOAD_SOILGRIDS=true

    for arg in "$@"; do
        case $arg in
            --wosis-only)
                DOWNLOAD_ERA5=false
                DOWNLOAD_SOILGRIDS=false
                ;;
            --era5-only)
                DOWNLOAD_WOSIS=false
                DOWNLOAD_SOILGRIDS=false
                ;;
            --soilgrids-only)
                DOWNLOAD_WOSIS=false
                DOWNLOAD_ERA5=false
                ;;
        esac
    done

    # Execute downloads
    if [ "$DOWNLOAD_WOSIS" = true ]; then
        download_wosis
    fi

    if [ "$DOWNLOAD_ERA5" = true ]; then
        download_era5
    fi

    if [ "$DOWNLOAD_SOILGRIDS" = true ]; then
        download_soilgrids
    fi

    echo ""
    echo "========================================"
    echo_info "All downloads complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run notebooks/data_exploration/01_wosis_exploration.ipynb"
    echo "  2. Verify data quality and coverage"
    echo "  3. Proceed to data integration notebook"
}

main "$@"

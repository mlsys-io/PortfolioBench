#!/usr/bin/env bash
# Download OHLCV data from Google Drive into user_data/data/.
#
# Datasets:
#   usstock     — crypto + US stocks + global indices (119 instruments x 3 timeframes)
#   polymarket  — Polymarket prediction-market contracts
#
# Requirements:
#   pip install gdown
#
# Usage:
#   bash utils/download_data.sh              # download both datasets
#   bash utils/download_data.sh usstock      # download only usstock
#   bash utils/download_data.sh polymarket   # download only polymarket

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/user_data/data"

# Google Drive folder IDs
USSTOCK_FOLDER_ID="18DqXyrfxDXxibC9gjm9TFzXolhaOBmyk"
POLYMARKET_FOLDER_ID="1x5jQ_8tkQhJuinhLKIctqa7aZ8D1uUHf"

# Ensure gdown is available
if ! command -v gdown &>/dev/null; then
    echo "Error: gdown is not installed. Install it with:"
    echo "  pip install gdown"
    exit 1
fi

download_folder() {
    local name="$1"
    local folder_id="$2"
    local dest="$DATA_DIR/$name"

    echo "----------------------------------------"
    echo "Downloading $name data into $dest ..."
    echo "----------------------------------------"
    mkdir -p "$dest"
    gdown --folder "https://drive.google.com/drive/folders/$folder_id" -O "$dest" --remaining-ok
    echo "Done: $(ls "$dest"/*.feather 2>/dev/null | wc -l) feather files in $dest"
    echo ""
}

DATASET="${1:-all}"

case "$DATASET" in
    usstock)
        download_folder "usstock" "$USSTOCK_FOLDER_ID"
        ;;
    polymarket)
        download_folder "polymarket" "$POLYMARKET_FOLDER_ID"
        ;;
    all)
        download_folder "usstock" "$USSTOCK_FOLDER_ID"
        download_folder "polymarket" "$POLYMARKET_FOLDER_ID"
        ;;
    *)
        echo "Usage: $0 [usstock|polymarket|all]"
        exit 1
        ;;
esac

echo "Data download complete."
echo ""
echo "To also generate data for the portfoliobench exchange (used by freqtrade backtesting),"
echo "copy the usstock files into the portfoliobench directory:"
echo "  mkdir -p $DATA_DIR/portfoliobench"
echo "  cp $DATA_DIR/usstock/*.feather $DATA_DIR/portfoliobench/"

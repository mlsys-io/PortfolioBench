#!/usr/bin/env python3
"""Download OHLCV feather data from Google Drive for PortfolioBench.

This script downloads the pre-built feather files (119 instruments x 3 timeframes
= 357 files) that power all PortfolioBench backtests. Data is stored as a single
compressed archive on Google Drive and extracted into the local data directories.

Usage:
    python utils/download_data.py                          # download all data
    python utils/download_data.py --exchange portfoliobench  # same as above
    python utils/download_data.py --output-dir /tmp/data   # custom output dir

Requirements:
    pip install gdown
"""

import argparse
import logging
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive file IDs for each exchange/dataset
# ---------------------------------------------------------------------------
GDRIVE_FILES = {
    "portfoliobench": {
        "file_id": "1BxEHMo5l8v7cZRqL-KQ2kP4d_Jxae5Vy",
        "archive_name": "usstock_data.tar.gz",
        "description": "Crypto + US Stocks + Global Indices (357 feather files)",
        "output_dirs": ["user_data/data/usstock", "user_data/data/portfoliobench"],
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_from_gdrive(file_id: str, output_path: str) -> bool:
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        logger.error(
            "gdown is required for downloading data. Install it with: pip install gdown"
        )
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    logger.info("Downloading from Google Drive (file_id=%s)...", file_id)

    try:
        gdown.download(url, output_path, quiet=False)
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            logger.info("Download complete: %s", output_path)
            return True
        logger.error("Download produced empty or missing file")
        return False
    except Exception as e:
        logger.error("Download failed: %s", e)
        return False


def extract_archive(archive_path: str, dest_dirs: list[str]) -> int:
    """Extract a tar.gz archive into one or more destination directories.

    Returns the number of feather files extracted.
    """
    count = 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Extracting %s...", archive_path)
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        # Find all feather files in the extracted tree
        tmp_path = Path(tmp_dir)
        feather_files = list(tmp_path.rglob("*.feather"))
        logger.info("Found %d feather files in archive", len(feather_files))

        if not feather_files:
            logger.error("No feather files found in archive")
            return 0

        for dest_dir in dest_dirs:
            dest_path = PROJECT_ROOT / dest_dir
            dest_path.mkdir(parents=True, exist_ok=True)

            for f in feather_files:
                target = dest_path / f.name
                shutil.copy2(f, target)
                count += 1

        logger.info("Extracted %d feather files to %s", count, ", ".join(dest_dirs))

    return count


def download_exchange_data(exchange: str, output_dir: str | None = None) -> bool:
    """Download and extract data for a given exchange."""
    if exchange not in GDRIVE_FILES:
        logger.error("Unknown exchange: %s (available: %s)", exchange, list(GDRIVE_FILES.keys()))
        return False

    config = GDRIVE_FILES[exchange]
    logger.info("Dataset: %s — %s", exchange, config["description"])

    # Determine output directories
    if output_dir:
        dest_dirs = [output_dir]
    else:
        dest_dirs = config["output_dirs"]

    # Download archive to temp file
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download_from_gdrive(config["file_id"], tmp_path):
            return False

        n_files = extract_archive(tmp_path, dest_dirs)
        if n_files == 0:
            return False

        logger.info("Successfully downloaded %d files for %s", n_files, exchange)
        return True
    finally:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download OHLCV feather data from Google Drive for PortfolioBench",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="portfoliobench",
        choices=list(GDRIVE_FILES.keys()),
        help="Exchange/dataset to download (default: portfoliobench)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory (overrides defaults)",
    )
    args = parser.parse_args()

    ok = download_exchange_data(args.exchange, args.output_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

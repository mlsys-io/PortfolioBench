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
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive folder IDs for each exchange/dataset
# ---------------------------------------------------------------------------
GDRIVE_FILES = {
    "portfoliobench": {
        "folder_id": "18DqXyrfxDXxibC9gjm9TFzXolhaOBmyk",
        "folder_url": "https://drive.google.com/drive/folders/18DqXyrfxDXxibC9gjm9TFzXolhaOBmyk",
        "archive_name": "usstock_data.tar.gz",
        "description": "Crypto + US Stocks + Global Indices (357 feather files)",
        "output_dirs": ["user_data/data/usstock", "user_data/data/portfoliobench"],
    },
    "polymarket": {
        "folder_id": "1x5jQ_8tkQhJuinhLKIctqa7aZ8D1uUHf",
        "folder_url": "https://drive.google.com/drive/folders/1x5jQ_8tkQhJuinhLKIctqa7aZ8D1uUHf",
        "archive_name": "polymarket_data.tar.gz",
        "description": "Polymarket prediction-market contracts",
        "output_dirs": ["user_data/data/polymarket"],
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_from_gdrive(folder_id: str, output_path: str, max_retries: int = 4) -> bool:
    """Download a folder from Google Drive using gdown.

    Retries with exponential backoff (2s, 4s, 8s, 16s) on failure to handle
    transient Google Drive rate-limiting and permission errors.
    """
    try:
        import gdown
    except ImportError:
        logger.error(
            "gdown is required for downloading data. Install it with: pip install gdown"
        )
        return False

    url = f"https://drive.google.com/drive/folders/{folder_id}"

    for attempt in range(1, max_retries + 1):
        logger.info(
            "Downloading from Google Drive folder (folder_id=%s) [attempt %d/%d]...",
            folder_id, attempt, max_retries,
        )

        try:
            # Clean output dir before each attempt to avoid partial state
            if os.path.isdir(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)

            gdown.download_folder(url, output=output_path, quiet=False)
            if os.path.isdir(output_path) and os.listdir(output_path):
                logger.info("Download complete: %s", output_path)
                return True
            logger.warning("Download produced empty or missing directory")
        except Exception as e:
            logger.warning("Download attempt %d failed: %s", attempt, e)

        if attempt < max_retries:
            wait = 2 ** attempt  # 2, 4, 8, 16 seconds
            logger.info("Retrying in %ds...", wait)
            time.sleep(wait)

    logger.error(
        "All %d download attempts failed for folder_id=%s", max_retries, folder_id
    )
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

    # Download folder to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = os.path.join(tmp_dir, "gdrive_download")

        if not download_from_gdrive(config["folder_id"], download_dir):
            return False

        # Check if downloaded files are an archive or direct feather files
        archive_path = None
        for f in Path(download_dir).rglob("*.tar.gz"):
            archive_path = str(f)
            break

        if archive_path:
            n_files = extract_archive(archive_path, dest_dirs)
        else:
            # Direct feather files downloaded from folder
            feather_files = list(Path(download_dir).rglob("*.feather"))
            if not feather_files:
                logger.error("No feather or archive files found in download")
                return False

            n_files = 0
            for dest_dir in dest_dirs:
                dest_path = PROJECT_ROOT / dest_dir
                dest_path.mkdir(parents=True, exist_ok=True)
                for f in feather_files:
                    shutil.copy2(f, dest_path / f.name)
                    n_files += 1

        if n_files == 0:
            return False

        logger.info("Successfully downloaded %d files for %s", n_files, exchange)
        return True


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

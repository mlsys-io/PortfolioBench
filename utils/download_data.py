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
        "file_id": "1BxEHMo5l8v7cZRqL-KQ2kP4d_Jxae5Vy",
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


def _download_file_direct(gdown_mod, file_id: str, output_path: str) -> bool:
    """Try downloading a single file directly by its Google Drive file ID.

    Uses gdown.download() with fuzzy=True and use_cookies=True to handle
    files that require confirmation (large files) or have restricted sharing
    settings that still allow cookie-based access.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(output_path, exist_ok=True)
    dest = os.path.join(output_path, "archive.tar.gz")

    logger.info("Trying direct file download (file_id=%s)...", file_id)
    try:
        result = gdown_mod.download(
            url, dest, quiet=False, fuzzy=True, use_cookies=True,
        )
        if result and os.path.isfile(dest) and os.path.getsize(dest) > 0:
            logger.info("Direct file download succeeded: %s", dest)
            return True
        logger.warning("Direct download returned no file or empty file")
    except Exception as e:
        logger.warning("Direct file download failed: %s", e)
    return False


def _download_folder(gdown_mod, folder_id: str, output_path: str) -> bool:
    """Download an entire Google Drive folder using gdown.download_folder()."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    logger.info("Trying folder download (folder_id=%s)...", folder_id)
    try:
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        gdown_mod.download_folder(url, output=output_path, quiet=False)
        if os.path.isdir(output_path) and os.listdir(output_path):
            logger.info("Folder download complete: %s", output_path)
            return True
        logger.warning("Folder download produced empty or missing directory")
    except Exception as e:
        logger.warning("Folder download failed: %s", e)
    return False


def download_from_gdrive(
    folder_id: str,
    output_path: str,
    max_retries: int = 4,
    file_id: str | None = None,
) -> bool:
    """Download data from Google Drive using gdown.

    Tries two strategies in order:
      1. Direct file download (if *file_id* is provided) — more reliable for
         large archives because it bypasses folder-listing permissions.
      2. Folder download — lists the folder contents and downloads each file.

    Each strategy is retried with exponential backoff (2s, 4s, 8s, 16s).
    """
    try:
        import gdown
    except ImportError:
        logger.error(
            "gdown is required for downloading data. Install it with: pip install gdown"
        )
        return False

    strategies: list[tuple[str, ...]] = []
    if file_id:
        strategies.append(("direct_file", file_id))
    strategies.append(("folder", folder_id))

    for strategy_name, gid in strategies:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "Download strategy=%s id=%s [attempt %d/%d]",
                strategy_name, gid, attempt, max_retries,
            )

            if strategy_name == "direct_file":
                ok = _download_file_direct(gdown, gid, output_path)
            else:
                ok = _download_folder(gdown, gid, output_path)

            if ok:
                return True

            if attempt < max_retries:
                wait = 2 ** attempt  # 2, 4, 8, 16 seconds
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

        logger.warning(
            "All %d attempts failed for strategy=%s id=%s",
            max_retries, strategy_name, gid,
        )

    logger.error("All download strategies exhausted for folder_id=%s", folder_id)
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

        if not download_from_gdrive(
            config["folder_id"], download_dir, file_id=config.get("file_id"),
        ):
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

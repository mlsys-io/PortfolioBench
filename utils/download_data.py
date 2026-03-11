#!/usr/bin/env python3
"""Download OHLCV feather data from Google Drive for PortfolioBench.

This script downloads the pre-built feather files (119 instruments x 3 timeframes
= 357 files) that power all PortfolioBench backtests.  Data is stored in a public
Google Drive **folder** and downloaded with ``gdown.download_folder``.

Usage:
    python utils/download_data.py                            # download all data
    python utils/download_data.py --exchange portfoliobench  # same as above
    python utils/download_data.py --output-dir /tmp/data     # custom output dir

Requirements:
    pip install gdown
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive IDs — the folder is the authoritative source; the archive
# file_id is kept as a fast-path but may not always be publicly accessible.
# ---------------------------------------------------------------------------
GDRIVE_FILES = {
    "portfoliobench": {
        "folder_id": "18DqXyrfxDXxibC9gjm9TFzXolhaOBmyk",
        "description": "Crypto + US Stocks + Global Indices (357 feather files)",
        "output_dir": "user_data/data/usstock",
        "expected_files": 357,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_folder(folder_id: str, dest_dir: Path, expected: int) -> int:
    """Download all files from a public Google Drive folder using gdown.

    Uses ``gdown.download_folder`` which fetches the folder listing once,
    then downloads each file.  ``remaining_ok=True`` means a partial
    download still succeeds (we count files afterwards).

    Returns the number of feather files placed in *dest_dir*.
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown is required: pip install gdown")
        return 0

    dest_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        logger.info("Downloading folder %s -> %s", folder_id, tmp_dir)

        try:
            gdown.download_folder(
                url,
                output=tmp_dir,
                quiet=False,
                remaining_ok=True,  # don't fail on partial download
            )
        except Exception as e:
            logger.warning("gdown.download_folder raised: %s", e)
            # Continue — there may already be partial results in tmp_dir

        # Collect all feather files from any sub-directory gdown created
        tmp_path = Path(tmp_dir)
        feather_files = list(tmp_path.rglob("*.feather"))
        logger.info("Downloaded %d / %d expected feather files", len(feather_files), expected)

        if not feather_files:
            logger.error("No feather files found after folder download")
            return 0

        # Move into the final destination
        for f in feather_files:
            target = dest_dir / f.name
            shutil.copy2(f, target)

    return len(list(dest_dir.glob("*.feather")))


def download_exchange_data(exchange: str, output_dir: str | None = None) -> bool:
    """Download data for a given exchange dataset."""
    if exchange not in GDRIVE_FILES:
        logger.error("Unknown exchange: %s (available: %s)", exchange, list(GDRIVE_FILES.keys()))
        return False

    config = GDRIVE_FILES[exchange]
    logger.info("Dataset: %s — %s", exchange, config["description"])

    dest = Path(output_dir) if output_dir else (PROJECT_ROOT / config["output_dir"])
    expected = config["expected_files"]

    # Check if data already exists (cache hit)
    existing = list(dest.glob("*.feather")) if dest.is_dir() else []
    if len(existing) >= expected:
        logger.info(
            "Data directory already has %d feather files (>= %d expected) — skipping download",
            len(existing),
            expected,
        )
        return True

    if existing:
        logger.info("Partial data found (%d / %d files) — downloading remaining", len(existing), expected)

    n_files = download_folder(config["folder_id"], dest, expected)

    if n_files == 0:
        logger.error("Download failed — no files obtained")
        return False

    if n_files < expected:
        logger.warning(
            "Partial download: got %d / %d files (Google Drive rate-limit). "
            "Re-run to fetch remaining files.",
            n_files,
            expected,
        )
        # Still return True — partial data is usable for benchmarks that
        # only need a subset of tickers.  The cache will persist what we
        # have, and the next run can pick up the rest.

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

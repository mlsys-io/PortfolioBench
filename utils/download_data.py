#!/usr/bin/env python3
"""Download OHLCV feather data from Google Drive for PortfolioBench.

Each Google Drive folder contains a single ``data.zip`` archive holding all
feather files for that dataset.  This script downloads the zip, extracts the
feather files into the local data directories, and cleans up the archive.

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
import re
import shutil
import sys
import tempfile
import time
import zipfile
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
        "description": "Crypto + US Stocks + Global Indices (357 feather files)",
        "output_dirs": ["user_data/data/usstock", "user_data/data/portfoliobench"],
        "min_expected_files": 357,
    },
    "polymarket": {
        "folder_id": "1x5jQ_8tkQhJuinhLKIctqa7aZ8D1uUHf",
        "folder_url": "https://drive.google.com/drive/folders/1x5jQ_8tkQhJuinhLKIctqa7aZ8D1uUHf",
        "description": "Polymarket prediction-market contracts",
        "output_dirs": ["user_data/data/polymarket"],
    },
    "kalshi": {
        "folder_id": "1DDo6uumqlsHeO4Ikvbo8LleEvWrXBKnP",
        "folder_url": "https://drive.google.com/drive/folders/1DDo6uumqlsHeO4Ikvbo8LleEvWrXBKnP",
        "description": "Kalshi prediction-market contracts",
        "output_dirs": ["user_data/data/kalshi"],
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _count_feather_files(directory: str) -> int:
    """Count .feather files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(".feather"))


def _is_valid_zip(filepath: str) -> bool:
    """Check if a file is a valid zip archive (magic bytes ``PK``)."""
    try:
        with open(filepath, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Download strategies — each tries to get the single data.zip from a folder.
# ---------------------------------------------------------------------------

def _download_via_gdown_folder(gdown_mod, folder_id: str, output_path: str) -> bool:
    """Download folder contents using ``gdown.download_folder()``."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    logger.info("Strategy: gdown folder download (folder_id=%s)...", folder_id)
    try:
        os.makedirs(output_path, exist_ok=True)
        gdown_mod.download_folder(
            url, output=output_path, quiet=False, remaining_ok=True,
        )
        # Check that at least one valid zip was downloaded
        for fname in os.listdir(output_path):
            if fname.endswith(".zip") and _is_valid_zip(os.path.join(output_path, fname)):
                logger.info("Folder download succeeded: found %s", fname)
                return True
        logger.warning("Folder download produced no valid zip files")
    except Exception as e:
        logger.warning("Folder download failed: %s", e)
    return False


def _download_via_api(gdown_mod, folder_id: str, output_path: str) -> bool:
    """List folder via Drive v3 API, then download the zip with gdown."""
    try:
        import requests
    except ImportError:
        logger.warning("requests not available for API strategy")
        return False

    logger.info("Strategy: API-based download (folder_id=%s)...", folder_id)

    # Fetch the folder page to extract the embedded API key
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        page_resp = session.get(folder_url, timeout=30)
        page_resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch folder page: %s", e)
        return False

    key_match = re.search(r'"key"\s*:\s*"(AIzaSy[A-Za-z0-9_-]+)"', page_resp.text)
    if not key_match:
        logger.warning("Could not extract API key from folder page")
        return False

    api_key = key_match.group(1)
    params: dict = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "pageSize": 100,
        "fields": "files(id,name)",
        "key": api_key,
    }
    try:
        resp = session.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            logger.warning("Drive API returned %d: %s", resp.status_code, resp.text[:300])
            return False
        files = resp.json().get("files", [])
    except Exception as e:
        logger.warning("Drive API request failed: %s", e)
        return False

    # Find the zip file entry
    zip_entry = next((f for f in files if f["name"].endswith(".zip")), None)
    if not zip_entry:
        logger.warning("No zip file found in API listing (%d files listed)", len(files))
        return False

    # Download via gdown using the file ID
    file_url = f"https://drive.google.com/uc?id={zip_entry['id']}"
    os.makedirs(output_path, exist_ok=True)
    dest = os.path.join(output_path, zip_entry["name"])
    logger.info("Downloading %s (id=%s)...", zip_entry["name"], zip_entry["id"])
    try:
        result = gdown_mod.download(file_url, dest, quiet=False, fuzzy=True, use_cookies=True)
        if result and _is_valid_zip(dest):
            logger.info("API-based download succeeded: %s", dest)
            return True
        logger.warning("API-based download produced invalid file")
        if os.path.isfile(dest):
            os.remove(dest)
    except Exception as e:
        logger.warning("gdown file download failed: %s", e)
    return False


def _download_via_requests(folder_id: str, output_path: str) -> bool:
    """List folder via Drive API, then download zip with pure requests."""
    try:
        import requests
    except ImportError:
        logger.warning("requests not available")
        return False

    logger.info("Strategy: pure-requests download (folder_id=%s)...", folder_id)

    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    try:
        page_resp = session.get(folder_url, timeout=30)
        page_resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch folder page: %s", e)
        return False

    key_match = re.search(r'"key"\s*:\s*"(AIzaSy[A-Za-z0-9_-]+)"', page_resp.text)
    if not key_match:
        logger.warning("Could not extract API key")
        return False

    api_key = key_match.group(1)
    params: dict = {
        "q": f"'{folder_id}' in parents and trashed = false",
        "pageSize": 100,
        "fields": "files(id,name)",
        "key": api_key,
    }
    try:
        resp = session.get(
            "https://www.googleapis.com/drive/v3/files",
            params=params,
            timeout=30,
        )
        if resp.status_code != 200:
            return False
        files = resp.json().get("files", [])
    except Exception:
        return False

    zip_entry = next((f for f in files if f["name"].endswith(".zip")), None)
    if not zip_entry:
        return False

    # Download with confirmation handling
    os.makedirs(output_path, exist_ok=True)
    dest = os.path.join(output_path, zip_entry["name"])
    base_url = "https://drive.google.com/uc"
    dl_params = {"id": zip_entry["id"], "export": "download"}

    try:
        resp = session.get(base_url, params=dl_params, stream=True, timeout=60)
        # Handle large-file confirmation
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                dl_params["confirm"] = value
                resp = session.get(base_url, params=dl_params, stream=True, timeout=60)
                break
        if resp.headers.get("content-type", "").startswith("text/html"):
            confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', resp.text)
            if confirm_match:
                dl_params["confirm"] = confirm_match.group(1)
                resp = session.get(base_url, params=dl_params, stream=True, timeout=60)
            else:
                logger.warning("HTML response without confirm token")
                return False
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        if _is_valid_zip(dest):
            logger.info("Requests-based download succeeded: %s", dest)
            return True
        logger.warning("Requests download produced invalid file")
        if os.path.isfile(dest):
            os.remove(dest)
    except Exception as e:
        logger.warning("Requests download failed: %s", e)
    return False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def download_folder_zip(folder_id: str, output_path: str, max_retries: int = 2) -> bool:
    """Download the data.zip from a Google Drive folder.

    Tries three strategies in order, each with up to *max_retries* attempts:
      1. ``gdown.download_folder()`` — simplest, works when gdown can see the folder.
      2. Drive API listing + ``gdown.download()`` — bypasses folder-listing issues.
      3. Drive API listing + pure ``requests`` download — no gdown dependency for
         the actual file transfer.
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown is required. Install it with: pip install gdown")
        return False

    strategies = [
        ("gdown_folder", lambda: _download_via_gdown_folder(gdown, folder_id, output_path)),
        ("api_gdown", lambda: _download_via_api(gdown, folder_id, output_path)),
        ("api_requests", lambda: _download_via_requests(folder_id, output_path)),
    ]

    for name, fn in strategies:
        for attempt in range(1, max_retries + 1):
            logger.info("Strategy=%s [attempt %d/%d]", name, attempt, max_retries)
            try:
                if fn():
                    return True
            except Exception as e:
                logger.warning("Strategy %s failed: %s", name, e)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)
        logger.warning("All %d attempts failed for strategy=%s", max_retries, name)

    logger.error("All download strategies exhausted for folder_id=%s", folder_id)
    return False


def extract_zip(zip_path: str, dest_dirs: list[str]) -> int:
    """Extract feather files from a zip archive into destination directories.

    Returns the number of unique feather files extracted.
    """
    if not zipfile.is_zipfile(zip_path):
        logger.error("Not a valid zip file: %s", zip_path)
        return 0

    unique_files: set[str] = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info("Extracting %s...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        feather_files = list(Path(tmp_dir).rglob("*.feather"))
        logger.info("Found %d feather files in archive", len(feather_files))

        if not feather_files:
            logger.error("No feather files found in archive")
            return 0

        for dest_dir in dest_dirs:
            dest_path = PROJECT_ROOT / dest_dir
            dest_path.mkdir(parents=True, exist_ok=True)
            for f in feather_files:
                shutil.copy2(f, dest_path / f.name)
                unique_files.add(f.name)

        logger.info(
            "Extracted %d unique feather files to %s",
            len(unique_files), ", ".join(dest_dirs),
        )

    return len(unique_files)


def download_exchange_data(exchange: str, output_dir: str | None = None) -> bool:
    """Download and extract data for a given exchange."""
    if exchange not in GDRIVE_FILES:
        logger.error("Unknown exchange: %s (available: %s)", exchange, list(GDRIVE_FILES.keys()))
        return False

    config = GDRIVE_FILES[exchange]
    logger.info("Dataset: %s -- %s", exchange, config["description"])

    # Determine output directories
    if output_dir:
        dest_dirs = [output_dir]
    else:
        dest_dirs = config["output_dirs"]

    min_expected = config.get("min_expected_files", 0)

    # Check if data already exists and is complete
    if not output_dir:
        first_dir = PROJECT_ROOT / dest_dirs[0]
        existing_count = _count_feather_files(str(first_dir))
        if min_expected > 0 and existing_count >= min_expected:
            logger.info(
                "Data already present: %d feather files in %s (expected >= %d). "
                "Skipping download.",
                existing_count, first_dir, min_expected,
            )
            return True
        elif existing_count > 0:
            logger.info(
                "Partial data found: %d feather files in %s (expected >= %d). "
                "Will re-download.",
                existing_count, first_dir, min_expected,
            )

    # Download zip to a temp directory, then extract
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = os.path.join(tmp_dir, "gdrive_download")

        if not download_folder_zip(config["folder_id"], download_dir):
            return False

        # Find the downloaded zip
        zip_path = None
        for fname in os.listdir(download_dir):
            fpath = os.path.join(download_dir, fname)
            if fname.endswith(".zip") and _is_valid_zip(fpath):
                zip_path = fpath
                break

        if not zip_path:
            logger.error("No valid zip archive found in download directory")
            return False

        # Extract feather files
        try:
            n_unique = extract_zip(zip_path, dest_dirs)
        except Exception as e:
            logger.error("Failed to extract archive: %s", e)
            return False

    if n_unique == 0:
        return False

    if min_expected > 0 and n_unique < min_expected:
        logger.warning(
            "Download incomplete: got %d feather files, expected >= %d.",
            n_unique, min_expected,
        )
        return False

    logger.info(
        "Successfully downloaded %d unique feather files for %s",
        n_unique, exchange,
    )
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

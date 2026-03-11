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
import re
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
        "min_expected_files": 357,
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


def _count_feather_files(directory: str) -> int:
    """Count .feather files in a directory (non-recursive)."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if f.endswith(".feather"))


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


def _download_folder(
    gdown_mod, folder_id: str, output_path: str,
    min_expected: int = 0,
) -> bool:
    """Download an entire Google Drive folder using gdown.download_folder().

    Uses ``remaining_ok=True`` to avoid gdown's 50-file hard error.
    Returns False if the number of downloaded files is below *min_expected*,
    so the caller can fall through to a more capable strategy.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    logger.info("Trying folder download (folder_id=%s)...", folder_id)
    try:
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        gdown_mod.download_folder(
            url, output=output_path, quiet=False, remaining_ok=True,
        )
        n_files = _count_feather_files(output_path)
        if n_files == 0:
            logger.warning("Folder download produced no feather files")
            return False

        if min_expected > 0 and n_files < min_expected:
            logger.warning(
                "Folder download incomplete: got %d feather files, expected >= %d "
                "(gdown folder listing is limited to ~50 files per page)",
                n_files, min_expected,
            )
            return False

        logger.info("Folder download complete: %d feather files in %s", n_files, output_path)
        return True
    except Exception as e:
        logger.warning("Folder download failed: %s", e)
    return False


def _list_folder_files_via_api(folder_id: str) -> list[dict]:
    """List ALL files in a public Google Drive folder using the Drive v3 API.

    Fetches the Google Drive folder page to extract the embedded API key,
    then paginates through the files.list endpoint to retrieve every file
    entry in the folder.

    Returns a list of dicts with ``id`` and ``name`` keys.
    """
    import requests

    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    })

    logger.info("Fetching folder page to extract API key...")
    try:
        page_resp = session.get(folder_url, timeout=30)
        page_resp.raise_for_status()
    except Exception as e:
        logger.warning("Failed to fetch folder page: %s", e)
        return []

    key_match = re.search(r'"key"\s*:\s*"(AIzaSy[A-Za-z0-9_-]+)"', page_resp.text)
    if not key_match:
        logger.warning("Could not extract API key from folder page")
        return []

    api_key = key_match.group(1)
    logger.info("Extracted API key, listing folder contents via Drive API...")

    all_files: list[dict] = []
    page_token = None

    while True:
        params: dict = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "pageSize": 1000,
            "fields": "nextPageToken,files(id,name)",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = session.get(
                "https://www.googleapis.com/drive/v3/files",
                params=params,
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Drive API files.list returned %d: %s",
                    resp.status_code, resp.text[:300],
                )
                break

            data = resp.json()
            files = data.get("files", [])
            all_files.extend(files)
            logger.info("  Listed %d files (total so far: %d)", len(files), len(all_files))

            page_token = data.get("nextPageToken")
            if not page_token:
                break
        except Exception as e:
            logger.warning("Drive API request failed: %s", e)
            break

    return all_files


def _download_folder_via_api(
    gdown_mod, folder_id: str, output_path: str,
) -> bool:
    """Download files from a Google Drive folder using the Drive API for listing.

    Uses :func:`_list_folder_files_via_api` to get the complete list of files
    in the folder (bypassing gdown's ~50-file listing limit), then downloads
    each file individually via ``gdown.download()``.

    Already-downloaded files (by name) are skipped so this strategy can build
    on partial results from a previous strategy.
    """
    logger.info("Trying API-based folder download (folder_id=%s)...", folder_id)

    file_entries = _list_folder_files_via_api(folder_id)
    if not file_entries:
        logger.warning("API listing returned no files")
        return False

    feather_entries = [f for f in file_entries if f["name"].endswith(".feather")]
    logger.info(
        "API listed %d total files (%d feather files)",
        len(file_entries), len(feather_entries),
    )

    if not feather_entries:
        logger.warning("No feather files found in API listing")
        return False

    os.makedirs(output_path, exist_ok=True)

    # Skip files that were already downloaded by a previous strategy
    existing = set(os.listdir(output_path)) if os.path.isdir(output_path) else set()
    to_download = [f for f in feather_entries if f["name"] not in existing]

    logger.info(
        "%d feather files already present, %d remaining to download",
        len(existing & {f["name"] for f in feather_entries}),
        len(to_download),
    )

    downloaded = len(existing & {f["name"] for f in feather_entries})
    for i, entry in enumerate(to_download, 1):
        file_url = f"https://drive.google.com/uc?id={entry['id']}"
        dest = os.path.join(output_path, entry["name"])
        try:
            result = gdown_mod.download(
                file_url, dest, quiet=True, use_cookies=True,
            )
            if result and os.path.isfile(dest) and os.path.getsize(dest) > 0:
                downloaded += 1
            else:
                logger.warning("Failed to download %s", entry["name"])
        except Exception as exc:
            logger.warning("Error downloading %s: %s", entry["name"], exc)

        if i % 50 == 0:
            logger.info("  Progress: %d/%d files downloaded", i, len(to_download))

    logger.info("API download complete: %d/%d feather files", downloaded, len(feather_entries))
    return downloaded > 0


def _download_folder_individually(
    gdown_mod, folder_id: str, output_path: str,
) -> bool:
    """Download files from a Google Drive folder one-by-one.

    Uses ``gdown.download_folder(skip_download=True, remaining_ok=True)`` to
    list available file metadata (up to the first page of results), then
    downloads each file individually via ``gdown.download()``.  This avoids
    the 50-file crash while still retrieving as many files as the listing
    provides.

    Already-downloaded files (by name) are skipped.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    logger.info(
        "Trying individual-file folder download (folder_id=%s)...", folder_id,
    )
    try:
        os.makedirs(output_path, exist_ok=True)

        file_entries = gdown_mod.download_folder(
            url, output=output_path, quiet=False,
            remaining_ok=True, skip_download=True,
        )
        if not file_entries:
            logger.warning("No files listed in folder")
            return False

        logger.info("Listed %d files in folder, downloading individually...", len(file_entries))

        existing = set(os.listdir(output_path)) if os.path.isdir(output_path) else set()

        downloaded = len(existing)
        for entry in file_entries:
            fname = os.path.basename(entry.local_path if hasattr(entry, "local_path") else entry.path)
            if fname in existing:
                continue

            file_url = f"https://drive.google.com/uc?id={entry.id}"
            dest = entry.local_path if hasattr(entry, "local_path") else os.path.join(output_path, fname)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                result = gdown_mod.download(
                    file_url, dest, quiet=False, use_cookies=True,
                )
                if result and os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    downloaded += 1
                else:
                    logger.warning("Failed to download %s", fname)
            except Exception as exc:
                logger.warning("Error downloading %s: %s", fname, exc)

        if downloaded > 0:
            logger.info(
                "Individual download complete: %d/%d files", downloaded, len(file_entries),
            )
            return True
        logger.warning("Individual-file download produced no files")
    except Exception as e:
        logger.warning("Individual-file folder download failed: %s", e)
    return False


def download_from_gdrive(
    folder_id: str,
    output_path: str,
    max_retries: int = 4,
    file_id: str | None = None,
    min_expected_files: int = 0,
) -> bool:
    """Download data from Google Drive using gdown.

    Tries four strategies in order:
      1. Direct file download (if *file_id* is provided) -- most reliable for
         large archives because it bypasses folder-listing permissions.
      2. Folder download -- bulk-downloads folder contents (with
         ``remaining_ok=True`` to handle folders with >50 files).  Returns
         False when the number of files is below *min_expected_files*, so that
         subsequent strategies can fetch the remaining files.
      3. API-based folder download -- uses the Google Drive v3 API to list ALL
         files in the folder (with pagination), then downloads each file
         individually.  Builds on any files already present in *output_path*.
      4. Individual file download via gdown listing -- lists folder contents
         via gdown (limited to ~50 files) then downloads one-by-one.

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
    strategies.append(("folder_api", folder_id))
    strategies.append(("folder_individual", folder_id))

    for strategy_name, gid in strategies:
        for attempt in range(1, max_retries + 1):
            logger.info(
                "Download strategy=%s id=%s [attempt %d/%d]",
                strategy_name, gid, attempt, max_retries,
            )

            if strategy_name == "direct_file":
                ok = _download_file_direct(gdown, gid, output_path)
            elif strategy_name == "folder":
                ok = _download_folder(
                    gdown, gid, output_path,
                    min_expected=min_expected_files,
                )
            elif strategy_name == "folder_api":
                ok = _download_folder_via_api(gdown, gid, output_path)
            else:
                ok = _download_folder_individually(gdown, gid, output_path)

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

    Returns the number of unique feather files extracted.
    """
    unique_files: set[str] = set()

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
                "Will attempt to download missing files.",
                existing_count, first_dir, min_expected,
            )

    # Download folder to temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        download_dir = os.path.join(tmp_dir, "gdrive_download")

        if not download_from_gdrive(
            config["folder_id"],
            download_dir,
            file_id=config.get("file_id"),
            min_expected_files=min_expected,
        ):
            return False

        # Check if downloaded files are an archive or direct feather files
        archive_path = None
        for f in Path(download_dir).rglob("*.tar.gz"):
            archive_path = str(f)
            break

        if archive_path:
            n_unique = extract_archive(archive_path, dest_dirs)
        else:
            # Direct feather files downloaded from folder
            feather_files = list(Path(download_dir).rglob("*.feather"))
            if not feather_files:
                logger.error("No feather or archive files found in download")
                return False

            unique_names: set[str] = set()
            for dest_dir in dest_dirs:
                dest_path = PROJECT_ROOT / dest_dir
                dest_path.mkdir(parents=True, exist_ok=True)
                for f in feather_files:
                    shutil.copy2(f, dest_path / f.name)
                    unique_names.add(f.name)

            n_unique = len(unique_names)

        if n_unique == 0:
            return False

        if min_expected > 0 and n_unique < min_expected:
            logger.warning(
                "Download incomplete: got %d unique feather files, expected >= %d. "
                "Benchmark will fall back to synthetic data.",
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

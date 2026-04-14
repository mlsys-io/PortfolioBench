"""Tests for data file integrity — verifies feather files have correct schema."""

import glob
import os

import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data", "data", "usstock")
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

# Representative subset: one crypto, one stock, one index per timeframe
SAMPLE_FILES = [
    "BTC_USDT-1d.feather",
    "BTC_USDT-4h.feather",
    "BTC_USDT-5m.feather",
    "AAPL_USD-1d.feather",
    "DJI_USD-1d.feather",
]


def _feather_available():
    path = os.path.join(DATA_DIR, "BTC_USDT-1d.feather")
    if not os.path.isfile(path):
        return False
    # Check it's not a Git LFS pointer (small text file starting with "version")
    with open(path, "rb") as f:
        header = f.read(20)
    return not header.startswith(b"version ")


@pytest.mark.skipif(not _feather_available(), reason="Data files not available (run: portbench download-data --exchange portfoliobench)")
class TestDataIntegrity:
    @pytest.mark.parametrize("filename", SAMPLE_FILES)
    def test_required_columns_present(self, filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} not found")
        df = pd.read_feather(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"{filename} missing columns: {missing}"

    @pytest.mark.parametrize("filename", SAMPLE_FILES)
    def test_no_empty_files(self, filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} not found")
        df = pd.read_feather(path)
        assert len(df) > 0, f"{filename} is empty"

    @pytest.mark.parametrize("filename", SAMPLE_FILES)
    def test_close_prices_positive(self, filename):
        path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(path):
            pytest.skip(f"{filename} not found")
        df = pd.read_feather(path)
        assert (df["close"] > 0).all(), f"{filename} has non-positive close prices"

    def test_naming_convention_consistent(self):
        files = glob.glob(os.path.join(DATA_DIR, "*.feather"))
        for f in files:
            basename = os.path.basename(f)
            # Expected: {TICKER}_USDT-{tf}.feather (crypto) or {TICKER}_USD-{tf}.feather (stocks/indices)
            has_quote = "_USDT-" in basename or "_USD-" in basename
            assert has_quote, f"Unexpected naming: {basename}"
            assert basename.endswith(".feather"), f"Wrong extension: {basename}"

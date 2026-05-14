"""Tests for OrderbookAlpha."""

import numpy as np
import pandas as pd
import pytest

import alpha.OrderbookAlpha as ob_mod
from alpha.OrderbookAlpha import OrderbookAlpha, _lookup_token_id

PANTHERS_PAIR  = "WillTheCarolinaPanthersWinSuperBowYES20250430/USDC"
PANTHERS_TOKEN = "26875704435144560123124814164931171497339462799728449796809868985717551034984"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n=100, start="2025-10-15", freq="4h"):
    """Synthetic OHLCV frame in Polymarket probability range [0, 1]."""
    np.random.seed(42)
    close = np.clip(0.01 + np.cumsum(np.random.randn(n) * 0.002), 0.001, 0.99)
    return pd.DataFrame({
        "date":   pd.date_range(start, periods=n, freq=freq),
        "open":   close - 0.001,
        "high":   close + 0.002,
        "low":    close - 0.002,
        "close":  close,
        "volume": np.ones(n) * 100.0,
    })


def _make_feat_orderbook(tmp_path, token_id="TEST_TOKEN", n=300, start="2025-10-14",
                         imbalance_values=None):
    """Write a minimal feat_<token_id>.parquet into tmp_path/feat_orderbook/.

    Parameters
    ----------
    imbalance_values : array-like, optional
        Fixed imbalance_3 values (length n). Defaults to random uniform [-1, 1].
    """
    feat_dir = tmp_path / "feat_orderbook"
    feat_dir.mkdir(exist_ok=True)

    np.random.seed(7)
    times = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    imb = imbalance_values if imbalance_values is not None else np.random.uniform(-1, 1, n)

    df = pd.DataFrame({"snapshot_time": times, "imbalance_3": imb})
    path = feat_dir / f"feat_{token_id}.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture()
def patched_feat_dir(tmp_path):
    """Fixture: patch _FEATURE_DIR to a temp directory and restore after test."""
    original = ob_mod._FEATURE_DIR
    ob_mod._FEATURE_DIR = tmp_path / "feat_orderbook"
    yield tmp_path
    ob_mod._FEATURE_DIR = original


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOrderbookAlpha:

    def test_adds_ob_columns(self, patched_feat_dir):
        _make_feat_orderbook(patched_feat_dir)
        result = OrderbookAlpha(_make_ohlcv(), {"token_id": "TEST_TOKEN"}).process()
        assert "ob_imbalance"     in result.columns
        assert "ob_imbalance_ema" in result.columns

    def test_unknown_token_returns_nan(self):
        """Missing feat file → both columns are NaN, no crash."""
        result = OrderbookAlpha(_make_ohlcv(), {"token_id": "does_not_exist"}).process()
        assert result["ob_imbalance"].isna().all()
        assert result["ob_imbalance_ema"].isna().all()

    def test_empty_metadata_returns_nan(self):
        result = OrderbookAlpha(_make_ohlcv(), {}).process()
        assert result["ob_imbalance"].isna().all()

    def test_imbalance_values_are_correct(self, patched_feat_dir):
        """Candles fully inside the snapshot range should reflect real imbalance data."""
        # Use a constant imbalance so we know the expected value
        _make_feat_orderbook(patched_feat_dir, imbalance_values=np.full(300, 0.5))
        df = _make_ohlcv()  # starts 2025-10-15, snapshots start 2025-10-14 → all covered
        result = OrderbookAlpha(df, {"token_id": "TEST_TOKEN"}).process()
        # Every candle has a prior snapshot → imbalance should be 0.5 throughout
        assert (result["ob_imbalance"] == 0.5).all()

    def test_imbalance_in_range(self, patched_feat_dir):
        _make_feat_orderbook(patched_feat_dir)
        result = OrderbookAlpha(_make_ohlcv(), {"token_id": "TEST_TOKEN"}).process()
        assert result["ob_imbalance"].between(-1, 1).all()

    def test_ema_is_smoother_than_raw(self, patched_feat_dir):
        _make_feat_orderbook(patched_feat_dir)
        result = OrderbookAlpha(_make_ohlcv(), {"token_id": "TEST_TOKEN"}).process()
        assert result["ob_imbalance_ema"].std() <= result["ob_imbalance"].std()

    def test_no_future_leakage(self, patched_feat_dir):
        """Candles before the first snapshot should get 0.0, not a future value."""
        # Snapshots start 2025-11-01; candles start 2025-10-01 (4 weeks earlier)
        _make_feat_orderbook(patched_feat_dir, start="2025-11-01",
                             imbalance_values=np.full(300, 0.9))
        df = _make_ohlcv(n=80, start="2025-10-01")
        result = OrderbookAlpha(df, {"token_id": "TEST_TOKEN"}).process()
        pre_snapshot = result[result["date"] < pd.Timestamp("2025-11-01")]
        assert (pre_snapshot["ob_imbalance"] == 0.0).all()

    def test_lookup_known_pair(self):
        assert _lookup_token_id(PANTHERS_PAIR) == PANTHERS_TOKEN

    def test_lookup_unknown_pair(self):
        assert _lookup_token_id("BTC/USDT") == ""

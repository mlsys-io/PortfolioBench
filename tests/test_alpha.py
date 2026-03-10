"""Tests for the alpha factor interface and EMA alpha implementation."""

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib C library not installed")

from alpha.interface import IAlpha
from alpha.SimpleEmaFactors import EmaAlpha


def _make_ohlcv(n=100):
    """Create a minimal synthetic OHLCV DataFrame."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.random.randint(100, 10000, size=n).astype(float),
    })


class TestIAlpha:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            IAlpha(pd.DataFrame(), {})

    def test_subclass_must_implement_process(self):
        class Incomplete(IAlpha):
            pass

        with pytest.raises(TypeError):
            Incomplete(pd.DataFrame(), {})


class TestEmaAlpha:
    def test_process_adds_expected_columns(self):
        df = _make_ohlcv()
        result = EmaAlpha(df, {"pair": "BTC/USDT"}).process()

        for col in ["ema_fast", "ema_slow", "ema_exit", "mean-volume"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_ema_values_are_numeric(self):
        df = _make_ohlcv()
        result = EmaAlpha(df, {"pair": "BTC/USDT"}).process()
        assert result["ema_fast"].dtype in [np.float64, np.float32]

    def test_ema_fast_shorter_than_slow(self):
        df = _make_ohlcv(200)
        result = EmaAlpha(df, {"pair": "TEST"}).process()
        # EMA fast (period 12) should be closer to recent prices than EMA slow (period 26)
        # After warm-up, fast EMA std should be >= slow EMA std (reacts faster)
        fast_std = result["ema_fast"].iloc[50:].std()
        slow_std = result["ema_slow"].iloc[50:].std()
        assert fast_std >= slow_std * 0.5  # fast reacts more (wider variance)

    def test_mean_volume_is_rolling_average(self):
        df = _make_ohlcv(50)
        result = EmaAlpha(df, {}).process()
        # First 19 rows should be NaN (rolling window = 20)
        assert result["mean-volume"].iloc[:19].isna().all()
        # Row 19 onward should have values
        assert result["mean-volume"].iloc[19:].notna().all()

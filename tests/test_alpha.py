"""Tests for the alpha factor interface and all alpha implementations."""

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib C library not installed")

from alpha.interface import IAlpha
from alpha.SimpleEmaFactors import EmaAlpha
from alpha.RsiAlpha import RsiAlpha
from alpha.MacdAlpha import MacdAlpha
from alpha.BollingerAlpha import BollingerAlpha


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


class TestRsiAlpha:
    def test_process_adds_expected_columns(self):
        df = _make_ohlcv()
        result = RsiAlpha(df, {"pair": "BTC/USDT"}).process()
        for col in ["rsi", "rsi_signal", "rsi_overbought", "rsi_oversold", "mean-volume"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_bounded_0_100(self):
        df = _make_ohlcv(200)
        result = RsiAlpha(df, {"pair": "TEST"}).process()
        valid = result["rsi"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_overbought_oversold_flags(self):
        df = _make_ohlcv(200)
        result = RsiAlpha(df, {"pair": "TEST"}).process()
        # Flags should be binary (0 or 1)
        assert set(result["rsi_overbought"].unique()).issubset({0, 1})
        assert set(result["rsi_oversold"].unique()).issubset({0, 1})

    def test_rsi_signal_is_smoothed(self):
        df = _make_ohlcv(200)
        result = RsiAlpha(df, {"pair": "TEST"}).process()
        # Signal line (rolling mean of RSI) should be smoother than raw RSI
        rsi_std = result["rsi"].iloc[30:].std()
        sig_std = result["rsi_signal"].iloc[30:].std()
        assert sig_std <= rsi_std


class TestMacdAlpha:
    def test_process_adds_expected_columns(self):
        df = _make_ohlcv()
        result = MacdAlpha(df, {"pair": "BTC/USDT"}).process()
        for col in ["macd", "macd_signal", "macd_hist", "macd_hist_rising", "mean-volume"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_macd_hist_equals_diff(self):
        df = _make_ohlcv(200)
        result = MacdAlpha(df, {"pair": "TEST"}).process()
        valid = result.dropna(subset=["macd", "macd_signal", "macd_hist"])
        diff = valid["macd"] - valid["macd_signal"]
        np.testing.assert_allclose(valid["macd_hist"].values, diff.values, atol=1e-10)

    def test_macd_hist_rising_is_binary(self):
        df = _make_ohlcv(200)
        result = MacdAlpha(df, {"pair": "TEST"}).process()
        valid = result["macd_hist_rising"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_macd_values_are_numeric(self):
        df = _make_ohlcv()
        result = MacdAlpha(df, {"pair": "BTC/USDT"}).process()
        assert result["macd"].dtype in [np.float64, np.float32]


class TestBollingerAlpha:
    def test_process_adds_expected_columns(self):
        df = _make_ohlcv()
        result = BollingerAlpha(df, {"pair": "BTC/USDT"}).process()
        for col in ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pctb", "mean-volume"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_upper_above_lower(self):
        df = _make_ohlcv(200)
        result = BollingerAlpha(df, {"pair": "TEST"}).process()
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_middle_is_between_bands(self):
        df = _make_ohlcv(200)
        result = BollingerAlpha(df, {"pair": "TEST"}).process()
        valid = result.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()
        assert (valid["bb_middle"] <= valid["bb_upper"]).all()

    def test_bb_width_positive(self):
        df = _make_ohlcv(200)
        result = BollingerAlpha(df, {"pair": "TEST"}).process()
        valid = result["bb_width"].dropna()
        assert (valid >= 0).all()

    def test_pctb_near_0_5_for_mean(self):
        # When price equals the middle band, %B should be ~0.5
        df = _make_ohlcv(200)
        result = BollingerAlpha(df, {"pair": "TEST"}).process()
        valid = result["bb_pctb"].dropna()
        # Mean %B across a random walk should be roughly centered around 0.5
        assert 0.2 < valid.mean() < 0.8

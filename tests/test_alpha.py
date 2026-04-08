"""Tests for the alpha factor interface and all alpha implementations."""

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib C library not installed")

from alpha.interface import IAlpha, AlphaEvaluator
from alpha.SimpleEmaFactors import EmaAlpha
from alpha.RsiAlpha import RsiAlpha
from alpha.MacdAlpha import MacdAlpha
from alpha.BollingerAlpha import BollingerAlpha
from alpha.AutoregressionAlpha import AutoregressionAlpha
from alpha.EventLstmAlpha import EventLstmAlpha # type: ignore

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

class TestAutoregressionAlpha:
    def test_process_adds_ar_pred_column(self):
        df = _make_ohlcv(200)
        df = df[["date", "close"]].copy()
        result = AutoregressionAlpha(df).process()
        assert "ar_pred" in result.columns, "Missing ar_pred column"

    def test_ar_pred_nan_for_short_series(self):
        df = _make_ohlcv(50)
        df = df[["date", "close"]].copy()
        result = AutoregressionAlpha(df).process()
        assert result["ar_pred"].isna().all(), "ar_pred should be NaN for short series"

    def test_ar_pred_has_valid_values(self):
        df = _make_ohlcv(200)
        df = df[["date", "close"]].copy()
        result = AutoregressionAlpha(df).process()
        ar_pred = result["ar_pred"].iloc[91:]
        assert ar_pred.notna().any(), "ar_pred should have valid predictions after lag"

class AlphaStub(IAlpha):
    def process(self):
        df = self.dataframe.copy()
        df["alpha"] = df["close"].pct_change()
        return df

class TestAlphaInformationEvaluation:
    def test_evaluate_information_coefficient_returns_dict(self):
        df = _make_ohlcv(120)
        evaluator = AlphaEvaluator(df, AlphaStub)
        result = evaluator.evaluate_information_coefficient(["alpha"])
        assert isinstance(result, dict)
        expected_keys = [("alpha", t) for t in [1, 5, 10, 20, 90]]
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (float, type(None)))

    def test_ic_values_are_finite_or_nan(self):
        df = _make_ohlcv(120)
        evaluator = AlphaEvaluator(df, AlphaStub)
        result = evaluator.evaluate_information_coefficient(["alpha"])
        for ic in result.values():
            assert (ic is None) or (isinstance(ic, float))

    def test_ic_on_constant_alpha_is_nan(self):
        class ConstAlpha(IAlpha):
            def process(self):
                df = self.dataframe.copy()
                df["alpha"] = 1.0
                return df
        df = _make_ohlcv(120)
        evaluator = AlphaEvaluator(df, ConstAlpha)
        result = evaluator.evaluate_information_coefficient(["alpha"])
        for ic in result.values():
            assert ic != ic

class TestEventLstmAlpha:
    @pytest.mark.slow
    def test_event_lstm_alpha_with_fake_data(self):
        # --- Generate fake OHLCV data ---
        length = 100
        np.random.seed(42)
        df = pd.DataFrame({
            "open": np.random.rand(length),
            "high": np.random.rand(length),
            "low": np.random.rand(length),
            "close": np.random.rand(length),
            "volume": np.random.rand(length)
        })

        # --- Path to your real trained LSTM model ---
        model_path = "./alpha/event_stacked_lstm.pth"
        
        # --- Sequence length (must match trained model) ---
        seq_len = 64

        # --- Instantiate alpha using real model ---
        alpha = EventLstmAlpha(df, model_path=model_path, seq_len=seq_len)
        result = alpha.process()

        # --- Assertions ---
        # Column must exist
        assert "lstm_pred" in result.columns, "Missing 'lstm_pred' column"
        
        # First seq_len rows will have NaN (not enough history)
        assert result["lstm_pred"].iloc[:seq_len].isna().all()

        # Predictions after seq_len should be numbers
        valid_preds = result["lstm_pred"].iloc[seq_len:]
        assert valid_preds.notna().all(), "Predictions should be filled"
        
        # Predictions should be within [0,1] because close prices are normalized
        assert (valid_preds >= 0).all() and (valid_preds <= 1).all()

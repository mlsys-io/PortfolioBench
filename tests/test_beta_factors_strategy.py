"""Tests for beta_factors_model strategy."""

# to test this file, run the command on bash: python -m pytest tests/test_beta_factors_strategy.py -q

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.crypto_factors_regression.beta_factors_model import beta_factors_model


def _make_weekly_ohlcv(n: int = 20, start: str = "2025-01-05") -> pd.DataFrame:
    """Create synthetic weekly OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n, freq="7D", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 2.0)

    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": np.random.randint(100, 10000, size=n).astype(float),
        }
    )


def _make_marketcap_csv(tmp_path: Path) -> Path:
    """Create fake Bitcoin market cap CSV."""
    dates = pd.date_range("2024-12-30", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timeClose": dates,
            "marketCap": np.linspace(1_000_000_000, 1_500_000_000, len(dates)),
        }
    )
    path = tmp_path / "Bitcoin_marketcap.csv"
    df.to_csv(path, sep=";", index=False)
    return path


class DummyModel:
    """Minimal sklearn-like model stub."""

    def __init__(self, preds):
        self.preds = np.asarray(preds)

    def predict(self, X):
        n = len(X)
        if self.preds.ndim == 0:
            return np.full((n,), float(self.preds))
        if len(self.preds) >= n:
            return self.preds[:n]
        return np.resize(self.preds, n)


class DummyTrade:
    """Minimal trade stub for custom_exit tests."""

    def __init__(self, open_date_utc: datetime | None = None):
        self.open_date_utc = open_date_utc or datetime.now(timezone.utc)


class TestBetaFactorsModelHelpers:
    def test_get_model_caches_loaded_model(self, monkeypatch):
        strat = beta_factors_model({})
        dummy_model = DummyModel([0.1])

        calls = {"count": 0}

        def fake_load(path):
            calls["count"] += 1
            return dummy_model

        monkeypatch.setattr("strategy.crypto_factors_regression.beta_factors_model.joblib.load", fake_load)

        model1 = strat.get_model()
        model2 = strat.get_model()

        assert model1 is dummy_model
        assert model2 is dummy_model
        assert calls["count"] == 1


class TestBetaFactorsModelMarketCap:
    def test_load_market_cap_data_returns_dataframe(self, monkeypatch, tmp_path):
        strat = beta_factors_model({})
        csv_path = _make_marketcap_csv(tmp_path)

        fake_strategy_file = tmp_path / "beta_factors_model.py"
        fake_strategy_file.write_text("# dummy")

        class DummyResolvedPath(type(Path())):
            pass

        monkeypatch.setattr(
            "strategy.crypto_factors_regression.beta_factors_model.Path.resolve",
            lambda self: fake_strategy_file,
        )

        result = strat.load_market_cap_data()

        assert result is not None
        assert not result.empty
        assert "timeClose" in result.columns
        assert "marketCap" in result.columns
        assert "marketCap_shifted" in result.columns

    def test_load_market_cap_data_is_cached(self, monkeypatch):
        strat = beta_factors_model({})
        cached = pd.DataFrame({"x": [1]})
        strat.btc_cap = cached

        result = strat.load_market_cap_data()

        assert result is cached


class TestBetaFactorsModelIndicators:
    def test_populate_indicators_adds_expected_columns(self):
        strat = beta_factors_model({})

        marketcap = pd.DataFrame(
            {
                "timeClose": pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC").floor("D"),
                "marketCap": np.linspace(1_000_000_000, 1_500_000_000, 60),
            }
        )
        marketcap["marketCap_shifted"] = marketcap["marketCap"].shift(1)

        strat.load_market_cap_data = lambda: marketcap
        strat.get_model = lambda: DummyModel(np.full(20, 0.08))

        df = _make_weekly_ohlcv(20)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        expected_cols = [
            "week_end",
            "marketCap_shifted",
            "ret",
            "cmkt",
            "cmom",
            "csize",
            "csize_cmkt",
            "cmkt_2",
            "cmom_3",
            "pred_ret",
            "ml_signal",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_pred_ret_is_numeric_and_aligned(self):
        strat = beta_factors_model({})

        marketcap = pd.DataFrame(
            {
                "timeClose": pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC").floor("D"),
                "marketCap": np.linspace(1_000_000_000, 1_500_000_000, 60),
            }
        )
        marketcap["marketCap_shifted"] = marketcap["marketCap"].shift(1)

        preds = np.linspace(-0.1, 0.1, 20)
        strat.load_market_cap_data = lambda: marketcap
        strat.get_model = lambda: DummyModel(preds)

        df = _make_weekly_ohlcv(20)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        assert len(result["pred_ret"]) == len(df)
        assert result["pred_ret"].dtype in [np.float64, np.float32]

    def test_ml_signal_positive_negative_and_zero(self):
        strat = beta_factors_model({})

        marketcap = pd.DataFrame(
            {
                "timeClose": pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC").floor("D"),
                "marketCap": np.linspace(1_000_000_000, 1_500_000_000, 60),
            }
        )
        marketcap["marketCap_shifted"] = marketcap["marketCap"].shift(1)

        preds = np.array([0.10, -0.10, 0.00, 0.08, -0.08] * 4)
        strat.load_market_cap_data = lambda: marketcap
        strat.get_model = lambda: DummyModel(preds)

        df = _make_weekly_ohlcv(20)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        valid_signals = set(result["ml_signal"].unique())
        assert valid_signals.issubset({-1, 0, 1})

    def test_ml_signal_forces_zero_when_features_contain_zero(self):
        strat = beta_factors_model({})

        marketcap = pd.DataFrame(
            {
                "timeClose": pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC").floor("D"),
                "marketCap": np.linspace(1_000_000_000, 1_100_000_000, 20),
            }
        )
        marketcap["marketCap_shifted"] = marketcap["marketCap"].shift(1)

        strat.load_market_cap_data = lambda: marketcap
        strat.get_model = lambda: DummyModel(np.full(10, 0.5))

        df = _make_weekly_ohlcv(10)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        # Early rows should become zero because rolling / pct-change features are zero-filled
        assert (result["ml_signal"].iloc[:3] == 0).any()


class TestBetaFactorsModelEntriesAndExits:
    def test_populate_entry_trend_sets_enter_long(self):
        strat = beta_factors_model({})

        df = pd.DataFrame(
            {
                "ml_signal": [0, 1, 1, -1],
                "volume": [100, 100, 0, 100],
            }
        )

        result = strat.populate_entry_trend(df.copy(), metadata={"pair": "BTC/USDT"})

        assert "enter_long" in result.columns
        assert result["enter_long"].fillna(0).sum() == 1

    def test_populate_exit_trend_sets_exit_long(self):
        strat = beta_factors_model({})

        df = pd.DataFrame(
            {
                "ml_signal": [0, -1, 1, -1],
                "volume": [100, 100, 100, 0],
            }
        )

        result = strat.populate_exit_trend(df.copy(), metadata={"pair": "BTC/USDT"})

        assert "exit_long" in result.columns
        assert result["exit_long"].fillna(0).sum() == 1

    def test_no_entry_when_volume_zero(self):
        strat = beta_factors_model({})

        df = pd.DataFrame(
            {
                "ml_signal": [1, 1],
                "volume": [0, 0],
            }
        )

        result = strat.populate_entry_trend(df.copy(), metadata={"pair": "BTC/USDT"})
        assert result.get("enter_long", pd.Series([0, 0])).fillna(0).sum() == 0

    def test_no_exit_when_volume_zero(self):
        strat = beta_factors_model({})

        df = pd.DataFrame(
            {
                "ml_signal": [-1, -1],
                "volume": [0, 0],
            }
        )

        result = strat.populate_exit_trend(df.copy(), metadata={"pair": "BTC/USDT"})
        assert result.get("exit_long", pd.Series([0, 0])).fillna(0).sum() == 0


class TestBetaFactorsModelCustomExit:
    def test_custom_exit_returns_time_exit_after_hold_days(self):
        strat = beta_factors_model({})
        trade = DummyTrade(open_date_utc=datetime.now(timezone.utc) - timedelta(days=8))

        result = strat.custom_exit(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )

        assert result == "time_exit"

    def test_custom_exit_returns_none_before_hold_days(self):
        strat = beta_factors_model({})
        trade = DummyTrade(open_date_utc=datetime.now(timezone.utc) - timedelta(days=3))

        result = strat.custom_exit(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )

        assert result is None


class TestBetaFactorsModelParameters:
    def test_strategy_static_attributes(self):
        assert beta_factors_model.timeframe == "1w"
        assert beta_factors_model.can_short is False
        assert beta_factors_model.stoploss == -0.15
        assert beta_factors_model.startup_candle_count == 10

    def test_threshold_parameters_exist(self):
        strat = beta_factors_model({})
        assert hasattr(strat, "buy_threshold")
        assert hasattr(strat, "sell_threshold")
"""Tests for the standalone portfolio pipeline (portfolio/PortfolioManagement.py)."""

import os

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib C library not installed")

from portfolio.PortfolioManagement import (
    align_close_prices,
    backtest_portfolio,
    blend_strategy_weights,
    build_ema_position_series,
    calculate_ons_weights,
    compute_metrics,
    ema_cross_signals,
    equal_weight_allocation,
    load_pair_data,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "user_data", "data", "binance")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pair_data(pairs=("A", "B"), n=100):
    """Create synthetic pair data dict suitable for pipeline functions."""
    np.random.seed(0)
    data = {}
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    for pair in pairs:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(100, 10000, size=n).astype(float),
        })
        data[pair] = df
    return data


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class TestLoadPairData:
    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(DATA_DIR, "BTC_USDT-1d.feather")),
        reason="Data files not available (LFS not pulled)",
    )
    def test_loads_existing_pairs(self):
        result = load_pair_data(DATA_DIR, ["BTC/USDT"], "1d")
        assert "BTC/USDT" in result
        assert "close" in result["BTC/USDT"].columns

    def test_skips_missing_pairs(self, tmp_path):
        result = load_pair_data(str(tmp_path), ["FAKE/USDT"], "1d")
        assert len(result) == 0


class TestAlignClosePrices:
    def test_produces_aligned_matrix(self):
        pair_data = _make_pair_data(("X", "Y"), n=50)
        prices = align_close_prices(pair_data)
        assert list(prices.columns) == ["X", "Y"]
        assert prices.shape[0] == 50


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

class TestEmaCrossSignals:
    def _enriched_df(self, n=200):
        from alpha.SimpleEmaFactors import EmaAlpha

        pair_data = _make_pair_data(("P",), n=n)
        df = EmaAlpha(pair_data["P"].copy(), {"pair": "P"}).process()
        return df

    def test_adds_signal_columns(self):
        df = self._enriched_df()
        result = ema_cross_signals(df)
        assert "enter_long" in result.columns
        assert "exit_long" in result.columns

    def test_signals_are_binary(self):
        df = self._enriched_df()
        result = ema_cross_signals(df)
        assert set(result["enter_long"].unique()).issubset({0, 1})
        assert set(result["exit_long"].unique()).issubset({0, 1})


class TestBuildEmaPositionSeries:
    def test_position_toggles_correctly(self):
        df = pd.DataFrame({
            "enter_long": [0, 1, 0, 0, 0, 0],
            "exit_long":  [0, 0, 0, 0, 1, 0],
        })
        pos = build_ema_position_series(df)
        expected = [0, 1, 1, 1, 0, 0]
        assert list(pos) == expected


# ---------------------------------------------------------------------------
# ONS weights
# ---------------------------------------------------------------------------

class TestCalculateOnsWeights:
    def test_weights_shape(self):
        pair_data = _make_pair_data(("A", "B", "C"), n=50)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        assert weights.shape == prices.shape

    def test_weights_sum_close_to_one(self):
        pair_data = _make_pair_data(("A", "B"), n=30)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        row_sums = weights.sum(axis=1)
        # ONS targets 0.95 sum (5% cash reserve)
        assert (row_sums > 0.5).all()
        assert (row_sums <= 1.01).all()

    def test_weights_non_negative(self):
        pair_data = _make_pair_data(("A", "B"), n=30)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        assert (weights.values >= -0.01).all()


# ---------------------------------------------------------------------------
# Allocation & blending
# ---------------------------------------------------------------------------

class TestEqualWeightAllocation:
    def test_uniform_weights(self):
        w = equal_weight_allocation(["A", "B", "C", "D"])
        assert len(w) == 4
        assert all(abs(v - 0.25) < 1e-10 for v in w.values())


class TestBlendStrategyWeights:
    def test_blended_weights_sum_to_one(self):
        pairs = ["A", "B"]
        index = pd.date_range("2025-01-01", periods=10, freq="D")
        ons = pd.DataFrame(np.full((10, 2), 0.5), index=index, columns=pairs)
        ema_positions = {
            "A": pd.Series(np.ones(10), index=index),
            "B": pd.Series(np.zeros(10), index=index),
        }
        eq = {"A": 0.5, "B": 0.5}
        blended = blend_strategy_weights(ons, ema_positions, eq)
        row_sums = blended.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Backtest & metrics
# ---------------------------------------------------------------------------

class TestBacktestPortfolio:
    def test_returns_expected_columns(self):
        pair_data = _make_pair_data(("A", "B"), n=50)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)
        assert "date" in result.columns
        assert "portfolio_value" in result.columns
        assert "daily_return" in result.columns
        assert len(result) == 50

    def test_initial_value_matches_capital(self):
        pair_data = _make_pair_data(("A",), n=20)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(np.ones((20, 1)), index=prices.index, columns=prices.columns)
        result = backtest_portfolio(prices, weights, initial_capital=5000)
        assert abs(result["portfolio_value"].iloc[0] - 5000) < 1


class TestComputeMetrics:
    def test_metrics_keys(self):
        result = pd.DataFrame({
            "portfolio_value": [10000, 10100, 10050, 10200],
            "daily_return": [0.0, 0.01, -0.005, 0.015],
        })
        m = compute_metrics(result)
        assert "total_return_pct" in m
        assert "annualised_return_pct" in m
        assert "annualised_sharpe" in m
        assert "max_drawdown_pct" in m
        assert "n_bars" in m

    def test_total_return_correct(self):
        result = pd.DataFrame({
            "portfolio_value": [100, 110],
            "daily_return": [0.0, 0.1],
        })
        m = compute_metrics(result)
        assert abs(m["total_return_pct"] - 10.0) < 0.01

    def test_max_drawdown_negative(self):
        result = pd.DataFrame({
            "portfolio_value": [100, 90, 95, 80],
            "daily_return": [0.0, -0.1, 0.056, -0.158],
        })
        m = compute_metrics(result)
        assert m["max_drawdown_pct"] < 0


# ---------------------------------------------------------------------------
# End-to-end pipeline (with real data if available)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    @pytest.mark.skipif(
        not os.path.isfile(os.path.join(DATA_DIR, "BTC_USDT-1d.feather")),
        reason="Data files not available (LFS not pulled)",
    )
    def test_run_portfolio_completes(self):
        from portfolio.PortfolioManagement import run_portfolio

        result, weights, metrics = run_portfolio(
            pairs=["BTC/USDT", "ETH/USDT"],
            timeframe="1d",
            initial_capital=10000,
        )
        assert len(result) > 0
        assert "total_return_pct" in metrics
        assert weights.shape[1] == 2

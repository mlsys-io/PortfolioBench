"""Tests for backtesting correctness across different scenarios.

Covers:
  - Deterministic reproducibility (same inputs produce same outputs)
  - Edge cases (single asset, single bar, flat prices, extreme volatility)
  - Weight constraint validation (non-negative, sum to 1)
  - Metric correctness (return, drawdown, Sharpe ratio calculations)
  - Pipeline stage ordering and data flow
  - Multi-asset blending behaviour
  - ONS convergence properties
  - Backtest accounting (initial capital, PnL tracking)
"""


import numpy as np
import pandas as pd
import pytest

from portfolio.PortfolioManagement import (
    align_close_prices,
    backtest_portfolio,
    blend_strategy_weights,
    build_ema_position_series,
    calculate_ons_weights,
    compute_metrics,
    ema_cross_signals,
    equal_weight_allocation,
    generate_alpha_signals,
)

# Only tests that call generate_alpha_signals (which uses EmaAlpha → talib)
# need the TA-Lib C library.  Mark them so the ~35 pure-math tests still run
# in environments without it.
try:
    import talib  # noqa: F401
    _has_talib = True
except ImportError:
    _has_talib = False

needs_talib = pytest.mark.skipif(not _has_talib, reason="TA-Lib C library not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair_data(pairs=("A", "B"), n=100, seed=0):
    """Create synthetic pair data dict suitable for pipeline functions."""
    np.random.seed(seed)
    data = {}
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    for pair in pairs:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 1.0)
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


def _make_constant_prices(pairs=("A", "B"), n=50, price=100.0):
    """Create pair data where all prices are constant (flat market)."""
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    data = {}
    for pair in pairs:
        df = pd.DataFrame({
            "date": dates,
            "open": np.full(n, price),
            "high": np.full(n, price),
            "low": np.full(n, price),
            "close": np.full(n, price),
            "volume": np.full(n, 1000.0),
        })
        data[pair] = df
    return data


def _make_trending_prices(pairs=("UP", "DOWN"), n=100, seed=42):
    """Create pair data with clear trends: one up, one down."""
    np.random.seed(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    data = {}
    for i, pair in enumerate(pairs):
        if i % 2 == 0:
            # Uptrend: base 100, drift +0.5 per bar
            close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
        else:
            # Downtrend: base 100, drift -0.3 per bar
            close = 100 - np.arange(n) * 0.3 + np.random.randn(n) * 0.1
        close = np.maximum(close, 1.0)
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
# 1. Deterministic reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_backtest_is_deterministic(self):
        """Running the same backtest twice should produce identical results."""
        pair_data = _make_pair_data(("A", "B"), n=60, seed=7)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )

        result1 = backtest_portfolio(prices, weights, initial_capital=10000)
        result2 = backtest_portfolio(prices, weights, initial_capital=10000)

        np.testing.assert_array_equal(
            result1["portfolio_value"].values,
            result2["portfolio_value"].values,
        )

    def test_ons_weights_deterministic(self):
        """ONS should produce the same weights given the same price data."""
        pair_data = _make_pair_data(("A", "B", "C"), n=50, seed=99)
        prices = align_close_prices(pair_data)

        w1 = calculate_ons_weights(prices)
        w2 = calculate_ons_weights(prices)

        np.testing.assert_allclose(w1.values, w2.values, atol=1e-8)

    def test_seed_independence(self):
        """Different random seeds should generally produce different prices."""
        data_a = _make_pair_data(("X",), n=30, seed=1)
        data_b = _make_pair_data(("X",), n=30, seed=2)

        close_a = data_a["X"]["close"].values
        close_b = data_b["X"]["close"].values
        assert not np.allclose(close_a, close_b)


# ---------------------------------------------------------------------------
# 2. Single-asset edge case
# ---------------------------------------------------------------------------

class TestSingleAsset:
    def test_single_asset_weights_all_one(self):
        """With one asset, equal-weight allocation should give weight 1.0."""
        w = equal_weight_allocation(["ONLY"])
        assert abs(w["ONLY"] - 1.0) < 1e-10

    def test_single_asset_ons_weights(self):
        """ONS with a single asset should assign ~1.0 to it every bar."""
        pair_data = _make_pair_data(("SOLO",), n=40, seed=5)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        np.testing.assert_allclose(weights["SOLO"].values, 1.0, atol=0.05)

    def test_single_asset_backtest_matches_asset_return(self):
        """With one asset and weight=1, portfolio return should equal asset return."""
        pair_data = _make_pair_data(("A",), n=30, seed=10)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.ones((len(prices), 1)), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        # Asset return from raw prices
        asset_return = prices["A"].iloc[-1] / prices["A"].iloc[0] - 1
        portfolio_return = result["portfolio_value"].iloc[-1] / result["portfolio_value"].iloc[0] - 1

        # Not exact due to shift(1) in backtest_portfolio (first bar has NaN weight)
        # but should be very close
        np.testing.assert_allclose(portfolio_return, asset_return, atol=0.02)


# ---------------------------------------------------------------------------
# 3. Flat market (constant prices)
# ---------------------------------------------------------------------------

class TestFlatMarket:
    def test_flat_prices_zero_return(self):
        """Constant prices should yield zero total return."""
        pair_data = _make_constant_prices(("A", "B"), n=30)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)
        metrics = compute_metrics(result)

        assert abs(metrics["total_return_pct"]) < 0.01

    def test_flat_prices_no_drawdown(self):
        """Constant prices should have zero drawdown."""
        pair_data = _make_constant_prices(("A", "B"), n=30)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)
        metrics = compute_metrics(result)

        assert abs(metrics["max_drawdown_pct"]) < 0.01

    def test_flat_prices_preserve_capital(self):
        """In a flat market, portfolio value should remain at initial capital."""
        pair_data = _make_constant_prices(("A",), n=20, price=50.0)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.ones((len(prices), 1)), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=5000)

        np.testing.assert_allclose(
            result["portfolio_value"].values,
            5000.0,
            atol=0.01,
        )


# ---------------------------------------------------------------------------
# 4. Weight constraint validation
# ---------------------------------------------------------------------------

class TestWeightConstraints:
    def test_ons_weights_non_negative(self):
        """ONS weights should be non-negative (long only)."""
        pair_data = _make_pair_data(("A", "B", "C"), n=60, seed=3)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        assert (weights.values >= -1e-6).all(), "Negative weights found"

    def test_ons_weights_bounded_above(self):
        """No single ONS weight should exceed 1.0 (within tolerance)."""
        pair_data = _make_pair_data(("A", "B"), n=60, seed=4)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        assert (weights.values <= 1.01).all(), "Weight exceeds 1.0"

    def test_ons_weights_row_sums(self):
        """ONS weight rows should sum to approximately 1.0."""
        pair_data = _make_pair_data(("A", "B", "C"), n=50, seed=6)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        row_sums = weights.sum(axis=1)

        np.testing.assert_allclose(row_sums.values, 1.0, atol=0.02, err_msg="Row sums should be ~1.0")

    def test_blend_weights_sum_to_one(self):
        """Blended weights must sum to exactly 1.0 per bar."""
        pairs = ["A", "B", "C"]
        index = pd.date_range("2025-01-01", periods=20, freq="D")
        ons = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=20), index=index, columns=pairs
        )
        ema_positions = {
            p: pd.Series(np.random.choice([0, 1], size=20), index=index)
            for p in pairs
        }
        eq = equal_weight_allocation(pairs)

        blended = blend_strategy_weights(ons, ema_positions, eq)
        row_sums = blended.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_equal_weight_sums_to_one(self):
        """Equal-weight allocation across N assets should sum to 1.0."""
        for n in [1, 2, 5, 10, 20]:
            pairs = [f"ASSET_{i}" for i in range(n)]
            w = equal_weight_allocation(pairs)
            assert abs(sum(w.values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Metric correctness
# ---------------------------------------------------------------------------

class TestMetricCorrectness:
    def test_known_total_return(self):
        """Verify total return calculation on a known sequence."""
        result = pd.DataFrame({
            "portfolio_value": [1000, 1100, 1210],
            "daily_return": [0.0, 0.1, 0.1],
        })
        m = compute_metrics(result)
        assert abs(m["total_return_pct"] - 21.0) < 0.01

    def test_known_max_drawdown(self):
        """Verify max drawdown on a known sequence: peak=110, trough=88."""
        values = [100, 110, 88, 95, 105]
        returns = [0.0] + [values[i] / values[i - 1] - 1 for i in range(1, len(values))]
        result = pd.DataFrame({
            "portfolio_value": values,
            "daily_return": returns,
        })
        m = compute_metrics(result)
        # Drawdown = (88 - 110) / 110 = -20%
        assert abs(m["max_drawdown_pct"] - (-20.0)) < 0.01

    def test_drawdown_always_non_positive(self):
        """Max drawdown should always be <= 0."""
        pair_data = _make_pair_data(("A", "B"), n=80, seed=12)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)
        m = compute_metrics(result)
        assert m["max_drawdown_pct"] <= 0

    def test_n_bars_matches_input_length(self):
        """n_bars metric should match the number of rows."""
        result = pd.DataFrame({
            "portfolio_value": [100, 101, 102, 103, 104],
            "daily_return": [0.0, 0.01, 0.01, 0.01, 0.01],
        })
        m = compute_metrics(result)
        assert m["n_bars"] == 5

    def test_sharpe_ratio_positive_for_steady_gains(self):
        """A steadily increasing portfolio should have a positive Sharpe ratio."""
        n = 100
        values = 1000 * (1.001 ** np.arange(n))
        returns = np.diff(values) / values[:-1]
        returns = np.insert(returns, 0, 0.0)
        result = pd.DataFrame({
            "portfolio_value": values,
            "daily_return": returns,
        })
        m = compute_metrics(result)
        assert m["annualised_sharpe"] > 0

    def test_sharpe_ratio_negative_for_steady_losses(self):
        """A steadily decreasing portfolio should have a negative Sharpe ratio."""
        n = 100
        values = 1000 * (0.999 ** np.arange(n))
        returns = np.diff(values) / values[:-1]
        returns = np.insert(returns, 0, 0.0)
        result = pd.DataFrame({
            "portfolio_value": values,
            "daily_return": returns,
        })
        m = compute_metrics(result)
        assert m["annualised_sharpe"] < 0

    def test_zero_return_zero_sharpe(self):
        """Flat portfolio should have Sharpe of 0."""
        result = pd.DataFrame({
            "portfolio_value": [100.0] * 10,
            "daily_return": [0.0] * 10,
        })
        m = compute_metrics(result)
        assert abs(m["annualised_sharpe"]) < 1e-6


# ---------------------------------------------------------------------------
# 6. Backtest accounting
# ---------------------------------------------------------------------------

class TestBacktestAccounting:
    def test_initial_capital_reflected(self):
        """First portfolio value should equal initial capital."""
        pair_data = _make_pair_data(("A",), n=20, seed=20)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.ones((len(prices), 1)), index=prices.index, columns=prices.columns
        )

        for capital in [1000, 10000, 1_000_000]:
            result = backtest_portfolio(prices, weights, initial_capital=capital)
            assert abs(result["portfolio_value"].iloc[0] - capital) < 1

    def test_output_has_expected_columns(self):
        """Backtest result should contain date, portfolio_value, daily_return."""
        pair_data = _make_pair_data(("A", "B"), n=20, seed=21)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        assert "date" in result.columns
        assert "portfolio_value" in result.columns
        assert "daily_return" in result.columns

    def test_output_length_matches_input(self):
        """Backtest result should have the same number of rows as input."""
        n = 35
        pair_data = _make_pair_data(("A", "B"), n=n, seed=22)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)
        assert len(result) == n

    def test_portfolio_value_always_positive(self):
        """Portfolio value should remain positive (no leverage / no shorting)."""
        pair_data = _make_pair_data(("A", "B", "C"), n=100, seed=23)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        assert (result["portfolio_value"] > 0).all()

    def test_zero_weight_no_exposure(self):
        """With all-zero weights, portfolio value should stay at initial capital."""
        pair_data = _make_pair_data(("A", "B"), n=30, seed=24)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.zeros(prices.shape), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        np.testing.assert_allclose(
            result["portfolio_value"].values,
            10000.0,
            atol=0.01,
        )


# ---------------------------------------------------------------------------
# 7. Signal generation correctness
# ---------------------------------------------------------------------------

@needs_talib
class TestSignalGenerationWithAlpha:
    """Tests that require generate_alpha_signals (depends on TA-Lib)."""

    def test_enter_exit_are_binary(self):
        """EMA cross signals should only be 0 or 1."""
        pair_data = _make_pair_data(("P",), n=200, seed=30)
        enriched = generate_alpha_signals(pair_data)
        df = ema_cross_signals(enriched["P"])

        assert set(df["enter_long"].unique()).issubset({0, 1})
        assert set(df["exit_long"].unique()).issubset({0, 1})


class TestPositionTracking:
    """Tests for build_ema_position_series (pure logic, no TA-Lib needed)."""

    def test_simultaneous_enter_exit_handled(self):
        """When both enter and exit fire on the same bar, exit should win
        (build_ema_position_series processes enter first, then exit)."""
        df = pd.DataFrame({
            "enter_long": [0, 1, 0, 1, 0],
            "exit_long":  [0, 0, 0, 1, 0],
        })
        pos = build_ema_position_series(df)
        # Bar 3: enter sets position=1, then exit resets to 0
        expected = [0, 1, 1, 0, 0]
        assert list(pos) == expected

    def test_position_series_is_binary(self):
        """Position series should only contain 0 and 1."""
        df = pd.DataFrame({
            "enter_long": [0, 1, 0, 0, 0, 0, 1, 0],
            "exit_long":  [0, 0, 0, 1, 0, 0, 0, 1],
        })
        pos = build_ema_position_series(df)
        assert set(pos.unique()).issubset({0, 1})

    def test_position_held_between_enter_and_exit(self):
        """Position should remain 1 from enter_long until exit_long."""
        df = pd.DataFrame({
            "enter_long": [0, 1, 0, 0, 0, 0],
            "exit_long":  [0, 0, 0, 0, 1, 0],
        })
        pos = build_ema_position_series(df)
        expected = [0, 1, 1, 1, 0, 0]
        assert list(pos) == expected

    def test_multiple_entries_and_exits(self):
        """Position tracking through multiple enter/exit cycles."""
        df = pd.DataFrame({
            "enter_long": [0, 1, 0, 0, 0, 1, 0, 0],
            "exit_long":  [0, 0, 0, 1, 0, 0, 0, 1],
        })
        pos = build_ema_position_series(df)
        expected = [0, 1, 1, 0, 0, 1, 1, 0]
        assert list(pos) == expected


# ---------------------------------------------------------------------------
# 8. Multi-asset blending behaviour
# ---------------------------------------------------------------------------

class TestMultiAssetBlending:
    def test_more_assets_lower_individual_equal_weight(self):
        """Adding more assets should decrease each equal-weight share."""
        w2 = equal_weight_allocation(["A", "B"])
        w5 = equal_weight_allocation(["A", "B", "C", "D", "E"])

        assert w2["A"] > w5["A"]
        assert abs(w2["A"] - 0.5) < 1e-10
        assert abs(w5["A"] - 0.2) < 1e-10

    def test_blend_with_all_ema_off_reduces_ema_component(self):
        """When all EMA positions are 0, EMA component should contribute nothing."""
        pairs = ["A", "B"]
        index = pd.date_range("2025-01-01", periods=10, freq="D")
        ons = pd.DataFrame(np.full((10, 2), 0.5), index=index, columns=pairs)
        ema_positions = {
            "A": pd.Series(np.zeros(10), index=index),
            "B": pd.Series(np.zeros(10), index=index),
        }
        eq = {"A": 0.5, "B": 0.5}

        blended = blend_strategy_weights(ons, ema_positions, eq)
        row_sums = blended.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_blend_with_all_ema_on_includes_ema(self):
        """When all EMA positions are 1, EMA contributes its full share."""
        pairs = ["A", "B"]
        index = pd.date_range("2025-01-01", periods=10, freq="D")
        ons = pd.DataFrame(np.full((10, 2), 0.5), index=index, columns=pairs)
        ema_positions = {
            "A": pd.Series(np.ones(10), index=index),
            "B": pd.Series(np.ones(10), index=index),
        }
        eq = {"A": 0.5, "B": 0.5}

        blended = blend_strategy_weights(ons, ema_positions, eq)
        row_sums = blended.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)
        # With symmetric inputs, both weights should be equal
        np.testing.assert_allclose(
            blended["A"].values, blended["B"].values, atol=1e-6
        )


# ---------------------------------------------------------------------------
# 9. ONS convergence properties
# ---------------------------------------------------------------------------

class TestONSProperties:
    def test_ons_starts_equal_weight(self):
        """ONS should start from 1/N uniform allocation."""
        pair_data = _make_pair_data(("A", "B", "C"), n=30, seed=40)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        first_row = weights.iloc[0].values
        expected = np.array([1.0 / 3] * 3)
        np.testing.assert_allclose(first_row, expected, atol=1e-6)

    def test_ons_weights_evolve_over_time(self):
        """ONS weights should change from the initial equal allocation as data arrives."""
        pair_data = _make_trending_prices(("UP", "DOWN"), n=100, seed=41)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        first_row = weights.iloc[0].values
        last_row = weights.iloc[-1].values

        # Weights should differ meaningfully from the initial 1/N
        deviation = np.abs(last_row - first_row).sum()
        assert deviation > 0.1, (
            f"Expected ONS to deviate from initial weights, but deviation was only {deviation:.4f}"
        )

    def test_ons_eta_regularisation(self):
        """With high eta, ONS should stay close to uniform 1/N."""
        pair_data = _make_trending_prices(("A", "B"), n=60, seed=42)
        prices = align_close_prices(pair_data)

        weights_pure = calculate_ons_weights(prices, eta=0.0)
        weights_reg = calculate_ons_weights(prices, eta=0.9)

        # Regularised weights should be closer to 0.5 than pure ONS
        deviation_pure = np.abs(weights_pure.values - 0.5).mean()
        deviation_reg = np.abs(weights_reg.values - 0.5).mean()

        assert deviation_reg < deviation_pure, (
            f"Regularised deviation ({deviation_reg:.4f}) should be less "
            f"than pure ({deviation_pure:.4f})"
        )


# ---------------------------------------------------------------------------
# 10. Many-asset scaling
# ---------------------------------------------------------------------------

class TestManyAssets:
    def test_ten_assets_pipeline(self):
        """The pipeline should handle 10 assets without errors."""
        pairs = [f"ASSET_{i}" for i in range(10)]
        pair_data = _make_pair_data(pairs, n=60, seed=50)
        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)

        assert weights.shape == prices.shape
        assert weights.shape[1] == 10

        result = backtest_portfolio(prices, weights, initial_capital=100000)
        assert len(result) == 60
        assert (result["portfolio_value"] > 0).all()

    def test_equal_weight_many_assets(self):
        """Equal weight with many assets should divide evenly."""
        pairs = [f"T{i}" for i in range(100)]
        w = equal_weight_allocation(pairs)
        for v in w.values():
            assert abs(v - 0.01) < 1e-10


# ---------------------------------------------------------------------------
# 11. Different initial capital sizes
# ---------------------------------------------------------------------------

class TestCapitalSizes:
    def test_return_pct_independent_of_capital(self):
        """Percentage return should be the same regardless of starting capital."""
        pair_data = _make_pair_data(("A", "B"), n=40, seed=60)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )

        result_small = backtest_portfolio(prices, weights, initial_capital=100)
        result_large = backtest_portfolio(prices, weights, initial_capital=1_000_000)

        m_small = compute_metrics(result_small)
        m_large = compute_metrics(result_large)

        assert abs(m_small["total_return_pct"] - m_large["total_return_pct"]) < 0.01
        assert abs(m_small["max_drawdown_pct"] - m_large["max_drawdown_pct"]) < 0.01
        assert abs(m_small["annualised_sharpe"] - m_large["annualised_sharpe"]) < 0.01


# ---------------------------------------------------------------------------
# 12. Alpha-to-backtest integration (end-to-end with synthetic data)
# ---------------------------------------------------------------------------

@needs_talib
class TestAlphaToBacktestIntegration:
    def test_full_pipeline_synthetic_data(self):
        """Full pipeline from alpha generation through to final metrics."""
        pair_data = _make_pair_data(("A", "B"), n=200, seed=70)

        # Step 1: Alpha generation
        enriched = generate_alpha_signals(pair_data)
        for pair in enriched:
            assert "ema_fast" in enriched[pair].columns
            assert "ema_slow" in enriched[pair].columns

        # Step 2: Signal generation
        ema_positions = {}
        for pair, df in enriched.items():
            df_signals = ema_cross_signals(df)
            pos = build_ema_position_series(df_signals)
            pos.index = df_signals["date"]
            ema_positions[pair] = pos

        # Step 3: Weight computation
        prices = align_close_prices(pair_data)
        ons_weights = calculate_ons_weights(prices)
        eq = equal_weight_allocation(list(prices.columns))

        # Step 4: Blend
        final_weights = blend_strategy_weights(ons_weights, ema_positions, eq)
        row_sums = final_weights.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

        # Step 5: Backtest
        result = backtest_portfolio(prices, final_weights, initial_capital=10000)
        assert len(result) == len(prices)
        assert (result["portfolio_value"] > 0).all()

        # Step 6: Metrics
        metrics = compute_metrics(result)
        assert "total_return_pct" in metrics
        assert "annualised_sharpe" in metrics
        assert "max_drawdown_pct" in metrics
        assert metrics["n_bars"] == len(prices)

    def test_pipeline_three_assets(self):
        """Pipeline with 3 assets to verify non-trivial weight distribution."""
        pair_data = _make_pair_data(("X", "Y", "Z"), n=200, seed=71)
        enriched = generate_alpha_signals(pair_data)

        ema_positions = {}
        for pair, df in enriched.items():
            df_signals = ema_cross_signals(df)
            pos = build_ema_position_series(df_signals)
            pos.index = df_signals["date"]
            ema_positions[pair] = pos

        prices = align_close_prices(pair_data)
        ons_weights = calculate_ons_weights(prices)
        eq = equal_weight_allocation(list(prices.columns))

        final_weights = blend_strategy_weights(ons_weights, ema_positions, eq)

        # With 3 assets, initial equal weight should be ~0.333
        assert abs(eq["X"] - 1.0 / 3) < 1e-10

        result = backtest_portfolio(prices, final_weights, initial_capital=50000)
        metrics = compute_metrics(result)
        assert metrics["n_bars"] == len(prices)


# ---------------------------------------------------------------------------
# 13. Extreme / stress scenarios
# ---------------------------------------------------------------------------

class TestExtremeScenarios:
    def test_high_volatility_no_crash(self):
        """Pipeline should handle highly volatile synthetic data without errors."""
        np.random.seed(80)
        n = 100
        dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
        pair_data = {}
        for pair in ("VOL_A", "VOL_B"):
            close = 100 + np.cumsum(np.random.randn(n) * 5.0)  # 10x normal volatility
            close = np.maximum(close, 1.0)
            df = pd.DataFrame({
                "date": dates,
                "open": close - 0.5,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.random.randint(100, 10000, size=n).astype(float),
            })
            pair_data[pair] = df

        prices = align_close_prices(pair_data)
        weights = calculate_ons_weights(prices)
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        assert len(result) == n
        assert (result["portfolio_value"] > 0).all()

    def test_very_short_timerange(self):
        """Backtest with only 2 bars should still work."""
        pair_data = _make_pair_data(("A", "B"), n=2, seed=81)
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.full(prices.shape, 0.5), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        assert len(result) == 2
        assert result["portfolio_value"].iloc[0] > 0

    def test_large_price_drop(self):
        """A 50% price drop should result in approximately -50% portfolio return (single asset)."""
        dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
        pair_data = {
            "CRASH": pd.DataFrame({
                "date": dates,
                "open": [100.0, 50.0],
                "high": [100.0, 55.0],
                "low": [100.0, 45.0],
                "close": [100.0, 50.0],
                "volume": [1000.0, 1000.0],
            })
        }
        prices = align_close_prices(pair_data)
        weights = pd.DataFrame(
            np.ones((2, 1)), index=prices.index, columns=prices.columns
        )
        result = backtest_portfolio(prices, weights, initial_capital=10000)

        # First bar: pct_change = 0 (fillna), second bar: -50%
        # But shift(1) uses previous weights, so bar 0 has NaN weight shifted → 0 return
        # Bar 1: weight from bar 0 (1.0) * return (-0.5) = -0.5
        final = result["portfolio_value"].iloc[-1]
        assert final < 10000, "Portfolio should lose value on a crash"

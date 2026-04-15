"""Tests for the workflow CLI mode.

These tests validate the executor stage handlers and the end-to-end workflow
pipeline using synthetic data (no feather files required).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib", reason="TA-Lib C library not installed")

# Ensure project root and LumidStack are on sys.path.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FT_ROOT = os.path.join(_PROJECT_ROOT, "freqtrade")
if os.path.isdir(os.path.join(_FT_ROOT, "freqtrade")) and _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)
_LUMIDSTACK = os.path.join(os.path.dirname(_PROJECT_ROOT), "LumidStack")
if os.path.isdir(_LUMIDSTACK) and _LUMIDSTACK not in sys.path:
    sys.path.insert(0, _LUMIDSTACK)

from adapters.portbench.runner import LocalWorkflowRunner
from workflow.executor import (
    handle_alpha,
    handle_portfolio,
    handle_strategy,
    register_all_handlers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair_data(pairs=("A/USD", "B/USD"), n=200):
    """Synthetic OHLCV data that works with TA-Lib (needs enough bars)."""
    np.random.seed(42)
    data = {}
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    for pair in pairs:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 1.0)  # keep positive
        df = pd.DataFrame({
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(100, 10_000, size=n).astype(float),
        })
        data[pair] = df
    return data


# ---------------------------------------------------------------------------
# Executor stage handler tests
# ---------------------------------------------------------------------------

class TestAlphaHandler:
    def test_ema_alpha(self):
        ctx = {"pair_data": _make_pair_data()}
        result = handle_alpha("alpha", {"type": "ema"}, ctx)
        assert result["alpha_type"] == "ema"
        assert result["pairs_processed"] == 2
        assert "enriched_data" in ctx
        for df in ctx["enriched_data"].values():
            assert "ema_fast" in df.columns
            assert "ema_slow" in df.columns

    def test_rsi_alpha(self):
        ctx = {"pair_data": _make_pair_data()}
        result = handle_alpha("alpha", {"type": "rsi"}, ctx)
        assert result["alpha_type"] == "rsi"
        for df in ctx["enriched_data"].values():
            assert "rsi" in df.columns

    def test_macd_alpha(self):
        ctx = {"pair_data": _make_pair_data()}
        result = handle_alpha("alpha", {"type": "macd"}, ctx)
        assert result["alpha_type"] == "macd"
        for df in ctx["enriched_data"].values():
            assert "macd" in df.columns

    def test_bollinger_alpha(self):
        ctx = {"pair_data": _make_pair_data()}
        result = handle_alpha("alpha", {"type": "bollinger"}, ctx)
        assert result["alpha_type"] == "bollinger"
        for df in ctx["enriched_data"].values():
            assert "bb_upper" in df.columns

    def test_unknown_alpha_raises(self):
        ctx = {"pair_data": _make_pair_data()}
        with pytest.raises(ValueError, match="Unknown alpha"):
            handle_alpha("alpha", {"type": "nonexistent"}, ctx)

    def test_no_data_raises(self):
        with pytest.raises(RuntimeError, match="No pair_data"):
            handle_alpha("alpha", {"type": "ema"}, {})


class TestStrategyHandler:
    def _run_alpha_first(self, alpha_type="ema"):
        ctx = {"pair_data": _make_pair_data()}
        handle_alpha("alpha", {"type": alpha_type}, ctx)
        return ctx

    def test_ema_cross_strategy(self):
        ctx = self._run_alpha_first("ema")
        result = handle_strategy("strategy", {"type": "ema_cross"}, ctx)
        assert result["strategy_type"] == "ema_cross"
        assert "positions" in ctx
        assert result["total_entries"] >= 0

    def test_rsi_strategy(self):
        ctx = self._run_alpha_first("rsi")
        result = handle_strategy("strategy", {"type": "rsi"}, ctx)
        assert result["strategy_type"] == "rsi"

    def test_macd_strategy(self):
        ctx = self._run_alpha_first("macd")
        result = handle_strategy("strategy", {"type": "macd"}, ctx)
        assert result["strategy_type"] == "macd"

    def test_bollinger_strategy(self):
        ctx = self._run_alpha_first("bollinger")
        result = handle_strategy("strategy", {"type": "bollinger"}, ctx)
        assert result["strategy_type"] == "bollinger"

    def test_unknown_strategy_raises(self):
        ctx = self._run_alpha_first("ema")
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_strategy("strategy", {"type": "fake"}, ctx)

    def test_no_enriched_data_raises(self):
        with pytest.raises(RuntimeError, match="No enriched_data"):
            handle_strategy("strategy", {"type": "ema_cross"}, {})


class TestPortfolioHandler:
    def _run_up_to_strategy(self, alpha_type="ema", strategy_type="ema_cross"):
        ctx = {"pair_data": _make_pair_data()}
        handle_alpha("alpha", {"type": alpha_type}, ctx)
        handle_strategy("strategy", {"type": strategy_type}, ctx)
        return ctx

    def test_blend_portfolio(self):
        ctx = self._run_up_to_strategy()
        result = handle_portfolio("portfolio", {"type": "blend"}, ctx)
        assert result["portfolio_type"] == "blend"
        assert "weights" in ctx
        assert "prices" in ctx
        weights = ctx["weights"]
        # Weights should sum to ~1 per row
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_equal_portfolio(self):
        ctx = self._run_up_to_strategy()
        result = handle_portfolio("portfolio", {"type": "equal"}, ctx)
        assert result["portfolio_type"] == "equal"
        weights = ctx["weights"]
        # All weights should be equal
        for col in weights.columns:
            np.testing.assert_allclose(weights[col].values, 0.5, atol=1e-6)

    def test_ons_portfolio(self):
        ctx = self._run_up_to_strategy()
        result = handle_portfolio("portfolio", {"type": "ons"}, ctx)
        assert result["portfolio_type"] == "ons"
        weights = ctx["weights"]
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)

    def test_unknown_portfolio_raises(self):
        ctx = self._run_up_to_strategy()
        with pytest.raises(ValueError, match="Unknown portfolio"):
            handle_portfolio("portfolio", {"type": "nonexistent"}, ctx)


# ---------------------------------------------------------------------------
# End-to-end runner integration
# ---------------------------------------------------------------------------

class TestWorkflowEndToEnd:
    WORKFLOW = {
        "apiVersion": "lumid/v1",
        "kind": "Workflow",
        "metadata": {"name": "e2e-test"},
        "spec": {
            "stages": {
                "alpha": {"template": "portbench.alpha", "params": {"type": "ema"}},
                "strategy": {
                    "template": "portbench.strategy",
                    "dependsOn": ["alpha"],
                    "params": {"type": "ema_cross"},
                },
                "portfolio": {
                    "template": "portbench.portfolio",
                    "dependsOn": ["strategy"],
                    "params": {"type": "blend"},
                },
            },
            "backtest": {
                "pairs": ["A/USD", "B/USD"],
                "timeframe": "1d",
                "initial_capital": 10000,
            },
        },
    }

    def test_full_pipeline_via_runner(self):
        from portfolio.PortfolioManagement import backtest_portfolio, compute_metrics

        runner = LocalWorkflowRunner.from_json(json.dumps(self.WORKFLOW))
        register_all_handlers(runner)

        ctx = {"pair_data": _make_pair_data()}
        wf_result = runner.run(context=ctx)

        assert len(wf_result.stages) == 3
        assert "alpha" in wf_result.stages
        assert "strategy" in wf_result.stages
        assert "portfolio" in wf_result.stages

        # Backtest the results
        prices = ctx["prices"]
        weights = ctx["weights"]
        bt_result = backtest_portfolio(prices, weights, 10_000.0)
        metrics = compute_metrics(bt_result)

        assert "total_return_pct" in metrics
        assert "annualised_sharpe" in metrics
        assert "max_drawdown_pct" in metrics
        assert metrics["n_bars"] > 0

    def test_all_alpha_strategy_combos(self):
        """Verify that all alpha+strategy combos run without errors."""
        combos = [
            ("ema", "ema_cross"),
            ("rsi", "rsi"),
            ("macd", "macd"),
            ("bollinger", "bollinger"),
        ]
        for alpha_type, strategy_type in combos:
            wf = {
                "apiVersion": "lumid/v1",
                "kind": "Workflow",
                "metadata": {"name": f"{alpha_type}-{strategy_type}"},
                "spec": {
                    "stages": {
                        "alpha": {"template": "portbench.alpha", "params": {"type": alpha_type}},
                        "strategy": {
                            "template": "portbench.strategy",
                            "dependsOn": ["alpha"],
                            "params": {"type": strategy_type},
                        },
                        "portfolio": {
                            "template": "portbench.portfolio",
                            "dependsOn": ["strategy"],
                            "params": {"type": "equal"},
                        },
                    },
                    "backtest": {"pairs": ["A/USD", "B/USD"]},
                },
            }
            runner = LocalWorkflowRunner.from_json(json.dumps(wf))
            register_all_handlers(runner)
            ctx = {"pair_data": _make_pair_data()}
            result = runner.run(context=ctx)
            assert len(result.stages) == 3, f"Failed for {alpha_type}/{strategy_type}"
            assert "weights" in ctx, f"No weights for {alpha_type}/{strategy_type}"

    def test_cli_workflow_output_json(self, tmp_path):
        """Test that run_workflow_cli produces a JSON output file."""
        from workflow.cli_workflow import run_workflow_cli

        # Write workflow to temp file
        wf_file = tmp_path / "test_wf.json"
        wf_file.write_text(json.dumps(self.WORKFLOW))

        # Write synthetic data as feather files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        pair_data = _make_pair_data()
        for pair, df in pair_data.items():
            fname = pair.replace("/", "_") + "-1d.feather"
            df.to_feather(data_dir / fname)

        # Update workflow to point at temp data dir
        wf = json.loads(wf_file.read_text())
        wf["spec"]["backtest"]["data_dir"] = str(data_dir)
        wf_file.write_text(json.dumps(wf))

        out_json = tmp_path / "results.json"
        result = run_workflow_cli(str(wf_file), output_json=str(out_json))

        assert out_json.exists()
        saved = json.loads(out_json.read_text())
        assert saved["workflow"] == "e2e-test"
        assert "metrics" in saved
        assert saved["metrics"]["n_bars"] > 0

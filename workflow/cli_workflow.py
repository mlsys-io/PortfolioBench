"""CLI entry point for the LumidStack workflow mode.

Usage::

    portbench workflow path/to/workflow.json
    portbench workflow path/to/workflow.json --output-json results.json

The workflow JSON follows the ``lumid/v1`` API version with
``kind: Workflow``.  The runner resolves stage dependencies,
executes each pipeline step (alpha -> strategy -> portfolio) via the
:class:`LocalWorkflowRunner` from LumidStack, then backtests the
resulting portfolio weights using PortfolioBench's backtest engine.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root and dependencies are importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FT_ROOT = os.path.join(_PROJECT_ROOT, "freqtrade")
if os.path.isdir(os.path.join(_FT_ROOT, "freqtrade")) and _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)

# LumidStack lives alongside PortfolioBench in the parent directory or can
# be installed as a package.  Try the sibling-directory layout first.
_LUMIDSTACK_ROOT = os.path.join(os.path.dirname(_PROJECT_ROOT), "LumidStack")
if os.path.isdir(_LUMIDSTACK_ROOT) and _LUMIDSTACK_ROOT not in sys.path:
    sys.path.insert(0, _LUMIDSTACK_ROOT)

logger = logging.getLogger(__name__)


def run_workflow_cli(
    workflow_file: str,
    *,
    output_json: str | None = None,
) -> dict[str, Any]:
    """Load a workflow JSON, execute via LumidStack, and backtest the output.

    Parameters
    ----------
    workflow_file
        Path to the ``lumid/v1`` workflow JSON.
    output_json
        If given, write the backtest metrics to this path as JSON.

    Returns
    -------
    dict
        Backtest metrics dictionary.
    """
    from adapters.portbench.runner import LocalWorkflowRunner
    from workflow.executor import register_all_handlers
    from portfolio.PortfolioManagement import (
        load_pair_data,
        backtest_portfolio,
        compute_metrics,
    )

    path = Path(workflow_file)
    if not path.is_file():
        raise FileNotFoundError(f"Workflow file not found: {workflow_file}")

    print("=" * 60)
    print("PORTFOLIOBENCH — WORKFLOW MODE")
    print("=" * 60)
    print(f"  Workflow file : {path.resolve()}")
    print()

    # ── 1. Parse workflow ──────────────────────────────────────────────
    runner = LocalWorkflowRunner.from_file(path)
    register_all_handlers(runner)

    wf_name = runner.workflow.metadata.name or path.stem

    # Parse backtest config from the extra spec (domain-specific fields
    # stored alongside stages in the workflow JSON).
    extra = runner.extra_spec
    bt_cfg = extra.get("backtest", {})
    bt_pairs = bt_cfg.get("pairs", [])
    bt_timeframe = bt_cfg.get("timeframe", "1d")
    bt_capital = float(bt_cfg.get("initial_capital", 10_000))
    bt_data_dir = bt_cfg.get("data_dir")

    print(f"  Workflow name : {wf_name}")
    print(f"  Pairs         : {bt_pairs}")
    print(f"  Timeframe     : {bt_timeframe}")
    print(f"  Capital       : ${bt_capital:,.2f}")
    print()

    # ── 2. Load data into context ──────────────────────────────────────
    data_dir = bt_data_dir
    if data_dir is None:
        data_dir = os.path.join(_PROJECT_ROOT, "user_data", "data", "usstock")

    pairs = bt_pairs
    if not pairs:
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "MSFT/USD"]

    print("=" * 60)
    print("STEP 0 — Loading market data")
    print("=" * 60)
    pair_data = load_pair_data(data_dir, pairs, bt_timeframe)
    if not pair_data:
        raise RuntimeError(
            "No data loaded — check data_dir and pairs in the workflow backtest section."
        )

    context: dict[str, Any] = {"pair_data": pair_data}

    # ── 3. Execute workflow via LumidStack runner ──────────────────────
    print()
    print("=" * 60)
    print("EXECUTING WORKFLOW via LumidStack LocalWorkflowRunner")
    print("=" * 60)
    wf_result = runner.run(context=context)

    for sname, sres in wf_result.stages.items():
        print(f"  Stage {sname!r}: {sres.duration_s:.3f}s — {sres.data}")

    # ── 4. Backtest the portfolio weights ──────────────────────────────
    prices = context.get("prices")
    weights = context.get("weights")

    if prices is None or weights is None:
        raise RuntimeError(
            "Workflow did not produce prices and weights in context. "
            "Ensure the workflow has alpha, strategy, and portfolio stages."
        )

    print()
    print("=" * 60)
    print("BACKTESTING — PortfolioBench")
    print("=" * 60)

    result_df = backtest_portfolio(prices, weights, bt_capital)
    metrics = compute_metrics(result_df)

    print(f"\n{'=' * 60}")
    print("PORTFOLIO RESULTS")
    print(f"{'=' * 60}")
    print(f"  Workflow          : {wf_name}")
    print(f"  Alpha             : {context.get('alpha_type', 'n/a')}")
    print(f"  Strategy          : {context.get('strategy_type', 'n/a')}")
    print(f"  Portfolio         : {context.get('portfolio_type', 'n/a')}")
    print(f"  Initial capital   : ${bt_capital:,.2f}")
    print(f"  Final value       : ${result_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Total return      : {metrics['total_return_pct']:.2f}%")
    print(f"  Annualised return : {metrics['annualised_return_pct']:.2f}%")
    print(f"  Annualised Sharpe : {metrics['annualised_sharpe']:.4f}")
    print(f"  Max drawdown      : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Number of bars    : {metrics['n_bars']}")
    print(f"  Workflow time     : {wf_result.total_duration_s:.3f}s")

    # ── 5. Optionally write results to JSON ────────────────────────────
    full_result = {
        "workflow": wf_name,
        "alpha_type": context.get("alpha_type"),
        "strategy_type": context.get("strategy_type"),
        "portfolio_type": context.get("portfolio_type"),
        "pairs": pairs,
        "timeframe": bt_timeframe,
        "initial_capital": bt_capital,
        "metrics": metrics,
        "stage_durations": {
            name: round(sr.duration_s, 4) for name, sr in wf_result.stages.items()
        },
        "total_workflow_time_s": round(wf_result.total_duration_s, 4),
    }

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(full_result, indent=2))
        print(f"\n  Results written to {out_path.resolve()}")

    return full_result

#!/usr/bin/env python3
"""
benchmark.py — PortfolioBench comprehensive testing & benchmarking suite
========================================================================
Runs trading strategies, portfolio algorithms, the standalone portfolio pipeline,
and alpha factor smoke tests across multiple asset classes and timeframes.
Produces a formatted terminal report and an optional JSON export.

Usage:
    python benchmark.py                  # full benchmark (all strategies, all assets, all timeframes)
    python benchmark.py --quick          # quick smoke test (subset of strategies/timeframes)
    python benchmark.py --trading-only   # only trading strategies
    python benchmark.py --portfolio-only # only portfolio strategies
    python benchmark.py --export report.json   # also write JSON results
"""

import os
import sys
import json
import time
import traceback
import argparse
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Freqtrade imports (deferred to allow graceful failure)
# ---------------------------------------------------------------------------
try:
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.optimize.backtesting import Backtesting
    from freqtrade.enums import RunMode
    FREQTRADE_AVAILABLE = True
except ImportError as e:
    FREQTRADE_AVAILABLE = False
    FREQTRADE_IMPORT_ERR = str(e)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG_PATH = os.path.join(PROJECT_ROOT, "user_data", "config.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "user_data", "data", "usstock")

# Trading strategies (from strategy/)
TRADING_STRATEGIES = [
    "EmaCrossStrategy",
    "MacdAdxStrategy",
    "IchimokuCloudStrategy",
    "RsiBollingerStrategy",
    "StochasticCciStrategy",
    "MlpSpeculativeStrategy",
]

# Portfolio strategies (from user_data/strategies/)
PORTFOLIO_STRATEGIES = [
    "ONS_Portfolio",
    "InverseVolatilityPortfolio",
    "MinimumVariancePortfolio",
    "BestSingleAssetPortfolio",
    "ExponentialGradientPortfolio",
    "MaxSharpePortfolio",
    "RiskParityPortfolio",
]

# Asset universes
ASSET_UNIVERSES = {
    "crypto":  ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
    "stocks":  ["AAPL/USDT", "MSFT/USDT", "NVDA/USDT", "GOOG/USDT"],
    "indices": ["DJI/USDT", "FTSE/USDT", "GSPC/USDT"],
    "mixed":   ["BTC/USDT", "ETH/USDT", "AAPL/USDT", "MSFT/USDT", "DJI/USDT", "GSPC/USDT"],
}

# Timeframes with appropriate date ranges
TIMEFRAME_CONFIG = {
    "5m":  {"timerange": "20260101-20260108", "label": "5-Minute"},
    "4h":  {"timerange": "20260101-20260131", "label": "4-Hour"},
    "1d":  {"timerange": "20240101-20260131", "label": "Daily"},
}

# Quick-mode subset
QUICK_TRADING = ["EmaCrossStrategy", "MacdAdxStrategy"]
QUICK_PORTFOLIO = ["ONS_Portfolio", "MaxSharpePortfolio"]
QUICK_ASSETS = {"crypto": ASSET_UNIVERSES["crypto"]}
QUICK_TIMEFRAMES = {"5m": TIMEFRAME_CONFIG["5m"]}


# ============================================================================
# REPORT FORMATTING HELPERS
# ============================================================================

class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{Colors.RESET}"


def banner(text: str, width: int = 80) -> str:
    border = "═" * width
    pad = (width - len(text) - 2) // 2
    line = "║" + " " * pad + text + " " * (width - pad - len(text) - 2) + "║"
    return f"\n{_c('╔' + border + '╗', Colors.CYAN)}\n{_c(line, Colors.CYAN)}\n{_c('╚' + border + '╝', Colors.CYAN)}\n"


def section(text: str) -> str:
    return f"\n{_c('┌─ ' + text + ' ' + '─' * max(0, 74 - len(text)), Colors.BLUE)}"


def subsection(text: str) -> str:
    return f"{_c('│  ', Colors.BLUE)}{_c(text, Colors.BOLD)}"


def detail(label: str, value: str) -> str:
    return f"{_c('│  ', Colors.BLUE)}  {label:<24s} {value}"


def status_pass() -> str:
    return _c("PASS", Colors.GREEN)


def status_fail() -> str:
    return _c("FAIL", Colors.RED)


def status_skip() -> str:
    return _c("SKIP", Colors.YELLOW)


def format_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    color = Colors.GREEN if val >= 0 else Colors.RED
    return _c(f"{val:+.2f}%", color)


def format_sharpe(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if val >= 1.0:
        color = Colors.GREEN
    elif val >= 0:
        color = Colors.YELLOW
    else:
        color = Colors.RED
    return _c(f"{val:.4f}", color)


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}m {secs:.1f}s"


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_single_backtest(
    strategy_name: str,
    strategy_path: str,
    pairs: List[str],
    timeframe: str,
    timerange: str,
    wallet: float = 1_000_000,
) -> Dict[str, Any]:
    """
    Run a single freqtrade backtest and extract key metrics.
    Returns a dict with status, metrics, and timing.
    """
    result: Dict[str, Any] = {
        "strategy": strategy_name,
        "pairs": pairs,
        "timeframe": timeframe,
        "timerange": timerange,
        "status": "error",
        "error": None,
        "metrics": {},
        "duration_s": 0.0,
    }

    if not FREQTRADE_AVAILABLE:
        result["status"] = "skip"
        result["error"] = f"freqtrade not importable: {FREQTRADE_IMPORT_ERR}"
        return result

    t0 = time.time()
    try:
        args = {
            "config": [CONFIG_PATH],
            "strategy": strategy_name,
            "timerange": timerange,
            "timeframe": timeframe,
            "strategy_path": strategy_path,
            "pairs": pairs,
            "dry_run_wallet": wallet,
        }

        config = setup_optimize_configuration(args, RunMode.BACKTEST)
        bt = Backtesting(config)
        bt_results = bt.start()

        # Extract results from the backtest
        metrics = _extract_backtest_metrics(bt_results, strategy_name)
        result["metrics"] = metrics
        result["status"] = "pass"

        if bt.exchange:
            bt.exchange.close()

    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)

    result["duration_s"] = time.time() - t0
    return result


def _extract_backtest_metrics(bt_results: Any, strategy_name: str) -> Dict[str, Any]:
    """Pull key metrics from freqtrade backtest result structure."""
    metrics: Dict[str, Any] = {}

    if bt_results is None:
        return metrics

    try:
        # bt_results is a dict keyed by strategy name
        if isinstance(bt_results, dict) and strategy_name in bt_results:
            strat_result = bt_results[strategy_name]
        elif isinstance(bt_results, dict) and "strategy" in bt_results:
            strat_result = bt_results
        elif isinstance(bt_results, dict):
            # Try first key
            first_key = next(iter(bt_results), None)
            strat_result = bt_results.get(first_key, bt_results)
        else:
            return metrics

        # Navigate the nested result structure
        if isinstance(strat_result, dict):
            # Look for backtest_results -> metrics mapping
            for key in ["results_per_pair", "results"]:
                if key in strat_result:
                    break

            # Extract from the strategy result dict
            if "trade_count" in strat_result:
                metrics["trades"] = strat_result.get("trade_count", 0)
            if "profit_total" in strat_result:
                metrics["total_return_pct"] = round(strat_result["profit_total"] * 100, 2)
            if "profit_total_abs" in strat_result:
                metrics["profit_abs"] = round(strat_result["profit_total_abs"], 2)
            if "max_drawdown" in strat_result:
                metrics["max_drawdown_pct"] = round(strat_result["max_drawdown"] * 100, 2)
            if "sharpe" in strat_result:
                metrics["sharpe"] = round(strat_result["sharpe"], 4)
            if "win_rate" in strat_result:
                metrics["win_rate_pct"] = round(strat_result["win_rate"] * 100, 2)
            if "profit_factor" in strat_result:
                metrics["profit_factor"] = round(strat_result["profit_factor"], 4)

            # Trades from backtest_results
            if "trades" in strat_result:
                trades_df = strat_result["trades"]
                if hasattr(trades_df, "__len__"):
                    metrics.setdefault("trades", len(trades_df))

    except Exception:
        pass

    return metrics


# ============================================================================
# PORTFOLIO PIPELINE TEST
# ============================================================================

def run_portfolio_pipeline_test() -> Dict[str, Any]:
    """Run the standalone portfolio pipeline (portfolio/PortfolioManagement.py)."""
    result: Dict[str, Any] = {
        "test": "portfolio_pipeline",
        "status": "error",
        "error": None,
        "metrics": {},
        "duration_s": 0.0,
    }
    t0 = time.time()
    try:
        from portfolio.PortfolioManagement import run_portfolio
        _, _, metrics = run_portfolio(
            data_dir=DATA_DIR,
            pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"],
            timeframe="1d",
            initial_capital=10_000.0,
        )
        result["metrics"] = metrics
        result["status"] = "pass"
    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)
    result["duration_s"] = time.time() - t0
    return result


# ============================================================================
# ALPHA FACTOR SMOKE TEST
# ============================================================================

def run_alpha_smoke_test() -> Dict[str, Any]:
    """Smoke-test all alpha factor implementations (EMA, RSI, MACD, Bollinger)."""
    result: Dict[str, Any] = {
        "test": "alpha_factors",
        "status": "error",
        "error": None,
        "details": {},
        "duration_s": 0.0,
    }
    t0 = time.time()
    try:
        import pandas as pd
        from alpha.SimpleEmaFactors import EmaAlpha
        from alpha.RsiAlpha import RsiAlpha
        from alpha.MacdAlpha import MacdAlpha
        from alpha.BollingerAlpha import BollingerAlpha

        # Load one pair's data
        filepath = os.path.join(DATA_DIR, "BTC_USDT-1d.feather")
        df = pd.read_feather(filepath)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)

        all_missing = []

        # --- EmaAlpha ---
        enriched = EmaAlpha(df.copy(), metadata={"pair": "BTC/USDT"}).process()
        ema_cols = ["ema_fast", "ema_slow", "ema_exit", "mean-volume"]
        all_missing += [c for c in ema_cols if c not in enriched.columns]
        result["details"]["ema"] = {
            "rows": len(enriched),
            "columns_added": ema_cols,
            "ema_fast_last": round(float(enriched["ema_fast"].iloc[-1]), 2),
            "ema_slow_last": round(float(enriched["ema_slow"].iloc[-1]), 2),
        }

        # --- RsiAlpha ---
        enriched = RsiAlpha(df.copy(), metadata={"pair": "BTC/USDT"}).process()
        rsi_cols = ["rsi", "rsi_signal", "rsi_overbought", "rsi_oversold", "mean-volume"]
        all_missing += [c for c in rsi_cols if c not in enriched.columns]
        result["details"]["rsi"] = {
            "rows": len(enriched),
            "columns_added": rsi_cols,
            "rsi_last": round(float(enriched["rsi"].iloc[-1]), 2),
        }

        # --- MacdAlpha ---
        enriched = MacdAlpha(df.copy(), metadata={"pair": "BTC/USDT"}).process()
        macd_cols = ["macd", "macd_signal", "macd_hist", "macd_hist_rising", "mean-volume"]
        all_missing += [c for c in macd_cols if c not in enriched.columns]
        result["details"]["macd"] = {
            "rows": len(enriched),
            "columns_added": macd_cols,
            "macd_last": round(float(enriched["macd"].iloc[-1]), 2),
        }

        # --- BollingerAlpha ---
        enriched = BollingerAlpha(df.copy(), metadata={"pair": "BTC/USDT"}).process()
        bb_cols = ["bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pctb", "mean-volume"]
        all_missing += [c for c in bb_cols if c not in enriched.columns]
        result["details"]["bollinger"] = {
            "rows": len(enriched),
            "columns_added": bb_cols,
            "bb_upper_last": round(float(enriched["bb_upper"].iloc[-1]), 2),
            "bb_lower_last": round(float(enriched["bb_lower"].iloc[-1]), 2),
        }

        if all_missing:
            result["status"] = "fail"
            result["error"] = f"Missing columns: {all_missing}"
        else:
            result["status"] = "pass"

    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)
    result["duration_s"] = time.time() - t0
    return result


# ============================================================================
# DATA INTEGRITY CHECK
# ============================================================================

def run_data_integrity_check() -> Dict[str, Any]:
    """Verify data files exist and are well-formed for all expected assets."""
    result: Dict[str, Any] = {
        "test": "data_integrity",
        "status": "error",
        "error": None,
        "details": {},
        "duration_s": 0.0,
    }
    t0 = time.time()
    try:
        import pandas as pd
        from pathlib import Path

        data_path = Path(DATA_DIR)
        feather_files = sorted(data_path.glob("*.feather"))
        total = len(feather_files)
        valid = 0
        invalid = []
        assets = set()
        timeframes_found = set()

        for f in feather_files:
            parts = f.stem.rsplit("-", 1)
            if len(parts) != 2:
                invalid.append(f.name)
                continue

            pair_str, tf = parts
            assets.add(pair_str)
            timeframes_found.add(tf)

            try:
                df = pd.read_feather(f)
                required = {"date", "open", "high", "low", "close", "volume"}
                if not required.issubset(set(df.columns)):
                    invalid.append(f"{f.name} (missing cols)")
                    continue
                if len(df) < 10:
                    invalid.append(f"{f.name} (only {len(df)} rows)")
                    continue
                valid += 1
            except Exception as e:
                invalid.append(f"{f.name} ({e})")

        result["details"] = {
            "total_files": total,
            "valid_files": valid,
            "invalid_files": len(invalid),
            "unique_assets": len(assets),
            "timeframes": sorted(timeframes_found),
            "invalid_list": invalid[:10],  # cap at 10 for readability
        }
        result["status"] = "pass" if len(invalid) == 0 else "fail"
        if invalid:
            result["error"] = f"{len(invalid)} invalid files found"
    except Exception as e:
        result["status"] = "fail"
        result["error"] = str(e)
    result["duration_s"] = time.time() - t0
    return result


# ============================================================================
# MAIN BENCHMARK ORCHESTRATOR
# ============================================================================

def run_benchmark(
    include_trading: bool = True,
    include_portfolio: bool = True,
    quick: bool = False,
    export_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full benchmark suite and print a formatted report.
    """
    all_results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "mode": "quick" if quick else "full",
        "data_integrity": {},
        "alpha_smoke_test": {},
        "portfolio_pipeline": {},
        "trading_backtests": [],
        "portfolio_backtests": [],
        "summary": {},
    }

    total_t0 = time.time()
    pass_count = 0
    fail_count = 0
    skip_count = 0

    # Select strategies & configs based on mode
    trading_strats = QUICK_TRADING if quick else TRADING_STRATEGIES
    portfolio_strats = QUICK_PORTFOLIO if quick else PORTFOLIO_STRATEGIES
    asset_configs = QUICK_ASSETS if quick else ASSET_UNIVERSES
    tf_configs = QUICK_TIMEFRAMES if quick else TIMEFRAME_CONFIG

    # ------------------------------------------------------------------
    print(banner(f"PortfolioBench — {'Quick' if quick else 'Full'} Benchmark Suite"))
    print(f"  {_c('Started:', Colors.DIM)} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {_c('Mode:', Colors.DIM)}    {'Quick (smoke test)' if quick else 'Full (all strategies × assets × timeframes)'}")
    strat_count = 0
    if include_trading:
        strat_count += len(trading_strats)
    if include_portfolio:
        strat_count += len(portfolio_strats)
    test_combos = strat_count * len(asset_configs) * len(tf_configs)
    print(f"  {_c('Matrix:', Colors.DIM)}  {strat_count} strategies × {len(asset_configs)} asset classes × {len(tf_configs)} timeframes = {test_combos} backtests")
    print(f"  {_c('Plus:', Colors.DIM)}    data integrity + alpha smoke test + portfolio pipeline")

    # ==================== 1. DATA INTEGRITY ====================
    print(section("1. Data Integrity Check"))
    data_result = run_data_integrity_check()
    all_results["data_integrity"] = data_result

    if data_result["status"] == "pass":
        pass_count += 1
        print(detail("Status", status_pass()))
    else:
        fail_count += 1
        print(detail("Status", f"{status_fail()} — {data_result.get('error', '')}"))

    d = data_result.get("details", {})
    print(detail("Files scanned", str(d.get("total_files", "?"))))
    print(detail("Valid / Invalid", f"{d.get('valid_files', '?')} / {d.get('invalid_files', '?')}"))
    print(detail("Unique assets", str(d.get("unique_assets", "?"))))
    print(detail("Timeframes", ", ".join(d.get("timeframes", []))))
    print(detail("Duration", format_duration(data_result["duration_s"])))

    # ==================== 2. ALPHA SMOKE TEST ==================
    print(section("2. Alpha Factor Smoke Test"))
    alpha_result = run_alpha_smoke_test()
    all_results["alpha_smoke_test"] = alpha_result

    if alpha_result["status"] == "pass":
        pass_count += 1
        print(detail("Status", status_pass()))
        ad = alpha_result.get("details", {})
        for alpha_name in ["ema", "rsi", "macd", "bollinger"]:
            sub = ad.get(alpha_name, {})
            if sub:
                cols = ", ".join(sub.get("columns_added", []))
                print(detail(f"  {alpha_name.upper()}", f"{sub.get('rows', '?')} rows — {cols}"))
    else:
        fail_count += 1
        print(detail("Status", f"{status_fail()} — {alpha_result.get('error', '')}"))
    print(detail("Duration", format_duration(alpha_result["duration_s"])))

    # ==================== 3. PORTFOLIO PIPELINE ================
    print(section("3. Standalone Portfolio Pipeline"))
    pipe_result = run_portfolio_pipeline_test()
    all_results["portfolio_pipeline"] = pipe_result

    if pipe_result["status"] == "pass":
        pass_count += 1
        print(detail("Status", status_pass()))
        pm = pipe_result.get("metrics", {})
        print(detail("Total return", format_pct(pm.get("total_return_pct"))))
        print(detail("Annualised return", format_pct(pm.get("annualised_return_pct"))))
        print(detail("Sharpe ratio", format_sharpe(pm.get("annualised_sharpe"))))
        print(detail("Max drawdown", format_pct(pm.get("max_drawdown_pct"))))
        print(detail("Bars", str(pm.get("n_bars", "?"))))
    else:
        fail_count += 1
        print(detail("Status", f"{status_fail()} — {pipe_result.get('error', '')}"))
    print(detail("Duration", format_duration(pipe_result["duration_s"])))

    # ==================== 4. TRADING STRATEGY BACKTESTS ========
    if include_trading:
        print(section(f"4. Trading Strategy Backtests ({len(trading_strats)} strategies)"))
        _run_strategy_suite(
            strat_names=trading_strats,
            strat_path=os.path.join(PROJECT_ROOT, "strategy"),
            asset_configs=asset_configs,
            tf_configs=tf_configs,
            result_list=all_results["trading_backtests"],
            counters={"pass": 0, "fail": 0, "skip": 0},
        )
        for r in all_results["trading_backtests"]:
            if r["status"] == "pass":
                pass_count += 1
            elif r["status"] == "fail":
                fail_count += 1
            else:
                skip_count += 1

    # ==================== 5. PORTFOLIO STRATEGY BACKTESTS ======
    if include_portfolio:
        print(section(f"5. Portfolio Strategy Backtests ({len(portfolio_strats)} strategies)"))
        _run_strategy_suite(
            strat_names=portfolio_strats,
            strat_path=os.path.join(PROJECT_ROOT, "user_data", "strategies"),
            asset_configs=asset_configs,
            tf_configs=tf_configs,
            result_list=all_results["portfolio_backtests"],
            counters={"pass": 0, "fail": 0, "skip": 0},
        )
        for r in all_results["portfolio_backtests"]:
            if r["status"] == "pass":
                pass_count += 1
            elif r["status"] == "fail":
                fail_count += 1
            else:
                skip_count += 1

    # ==================== SUMMARY ==============================
    total_duration = time.time() - total_t0
    total_tests = pass_count + fail_count + skip_count
    all_results["summary"] = {
        "total": total_tests,
        "passed": pass_count,
        "failed": fail_count,
        "skipped": skip_count,
        "duration_s": round(total_duration, 2),
    }

    _print_summary_report(all_results, total_duration)

    # Export if requested
    if export_path:
        _export_json(all_results, export_path)

    return all_results


def _run_strategy_suite(
    strat_names: List[str],
    strat_path: str,
    asset_configs: Dict[str, List[str]],
    tf_configs: Dict[str, Dict],
    result_list: List[Dict],
    counters: Dict[str, int],
):
    """Run all strategy × asset × timeframe combinations and print results."""
    for strat in strat_names:
        print(subsection(f"Strategy: {strat}"))
        for asset_label, pairs in asset_configs.items():
            for tf, tf_cfg in tf_configs.items():
                tag = f"{asset_label}/{tf}"
                r = run_single_backtest(
                    strategy_name=strat,
                    strategy_path=strat_path,
                    pairs=pairs,
                    timeframe=tf,
                    timerange=tf_cfg["timerange"],
                )
                r["asset_class"] = asset_label
                result_list.append(r)

                if r["status"] == "pass":
                    counters["pass"] += 1
                    m = r.get("metrics", {})
                    ret_str = format_pct(m.get("total_return_pct"))
                    trades_str = str(m.get("trades", "?"))
                    dur_str = format_duration(r["duration_s"])
                    print(detail(f"[{status_pass()}] {tag}", f"return={ret_str}  trades={trades_str}  {dur_str}"))
                elif r["status"] == "fail":
                    counters["fail"] += 1
                    err = (r.get("error") or "")[:60]
                    print(detail(f"[{status_fail()}] {tag}", f"{err}  {format_duration(r['duration_s'])}"))
                else:
                    counters["skip"] += 1
                    print(detail(f"[{status_skip()}] {tag}", r.get("error", "")[:60]))


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def _print_summary_report(results: Dict[str, Any], total_duration: float):
    """Print the final formatted summary."""
    s = results["summary"]

    print(banner("Benchmark Results Summary"))

    # Overview bar
    total = s["total"]
    passed = s["passed"]
    failed = s["failed"]
    skipped = s["skipped"]

    bar_width = 50
    pass_w = int(bar_width * passed / max(total, 1))
    fail_w = int(bar_width * failed / max(total, 1))
    skip_w = bar_width - pass_w - fail_w

    bar = (
        _c("█" * pass_w, Colors.GREEN)
        + _c("█" * fail_w, Colors.RED)
        + _c("█" * skip_w, Colors.YELLOW)
    )
    print(f"  {bar}  {passed}/{total} passed")
    print()

    # Counts
    print(f"  {_c('Total tests:', Colors.BOLD)}   {total}")
    print(f"  {_c('Passed:', Colors.GREEN)}        {passed}")
    print(f"  {_c('Failed:', Colors.RED)}        {failed}")
    print(f"  {_c('Skipped:', Colors.YELLOW)}       {skipped}")
    print(f"  {_c('Duration:', Colors.DIM)}      {format_duration(total_duration)}")
    print()

    # Trading strategy leaderboard
    if results["trading_backtests"]:
        _print_leaderboard("Trading Strategy Leaderboard", results["trading_backtests"])

    # Portfolio strategy leaderboard
    if results["portfolio_backtests"]:
        _print_leaderboard("Portfolio Strategy Leaderboard", results["portfolio_backtests"])

    # Pipeline summary
    pm = results.get("portfolio_pipeline", {}).get("metrics", {})
    if pm:
        print(section("Standalone Pipeline"))
        print(detail("Total return", format_pct(pm.get("total_return_pct"))))
        print(detail("Sharpe", format_sharpe(pm.get("annualised_sharpe"))))
        print(detail("Max drawdown", format_pct(pm.get("max_drawdown_pct"))))
    print()

    # Final verdict
    if failed == 0 and skipped == 0:
        print(f"  {_c('✓ ALL TESTS PASSED', Colors.GREEN + Colors.BOLD)}")
    elif failed == 0:
        print(f"  {_c('✓ ALL RUN TESTS PASSED', Colors.GREEN + Colors.BOLD)} ({skipped} skipped)")
    else:
        print(f"  {_c(f'✗ {failed} TEST(S) FAILED', Colors.RED + Colors.BOLD)}")
    print()


def _print_leaderboard(title: str, backtest_results: List[Dict]):
    """Print a ranked table of strategies by return."""
    print(section(title))

    # Aggregate: best return per strategy
    strat_best: Dict[str, Dict] = {}
    for r in backtest_results:
        name = r["strategy"]
        ret = r.get("metrics", {}).get("total_return_pct")
        if ret is None:
            continue
        if name not in strat_best or ret > strat_best[name]["return"]:
            strat_best[name] = {
                "return": ret,
                "sharpe": r.get("metrics", {}).get("sharpe"),
                "trades": r.get("metrics", {}).get("trades", 0),
                "asset_class": r.get("asset_class", "?"),
                "timeframe": r.get("timeframe", "?"),
                "drawdown": r.get("metrics", {}).get("max_drawdown_pct"),
            }

    ranked = sorted(strat_best.items(), key=lambda x: x[1]["return"], reverse=True)

    # Table header
    hdr = f"  {'#':<4}{'Strategy':<35}{'Return':>10}{'Sharpe':>10}{'Trades':>8}{'DD':>10}{'Best On':>12}"
    print(_c(hdr, Colors.DIM))
    print(_c("  " + "─" * 85, Colors.DIM))

    for i, (name, info) in enumerate(ranked, 1):
        ret_s = format_pct(info["return"])
        sharpe_s = format_sharpe(info.get("sharpe"))
        trades_s = str(info.get("trades", "?"))
        dd_s = format_pct(info.get("drawdown"))
        best_s = f"{info['asset_class']}/{info['timeframe']}"
        print(f"  {i:<4}{name:<35}{ret_s:>10}{sharpe_s:>10}{trades_s:>8}{dd_s:>10}{best_s:>12}")

    print()

    # Pass/fail summary for this category
    passed = sum(1 for r in backtest_results if r["status"] == "pass")
    failed = sum(1 for r in backtest_results if r["status"] == "fail")
    total = len(backtest_results)
    print(detail("Results", f"{passed}/{total} passed, {failed} failed"))


# ============================================================================
# JSON EXPORT
# ============================================================================

def _export_json(results: Dict[str, Any], path: str):
    """Write results to a JSON file."""
    # Make results JSON-serializable
    def _clean(obj):
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return None
            return round(obj, 6)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    cleaned = _clean(results)
    with open(path, "w") as f:
        json.dump(cleaned, f, indent=2, default=str)

    print(f"  {_c('Exported:', Colors.DIM)} {path}")


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PortfolioBench — Comprehensive Testing & Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                     Full benchmark suite
  python benchmark.py --quick             Quick smoke test
  python benchmark.py --trading-only      Only trading strategies
  python benchmark.py --portfolio-only    Only portfolio strategies
  python benchmark.py --export report.json  Export results to JSON
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (subset of strategies/timeframes)")
    parser.add_argument("--trading-only", action="store_true", help="Only run trading strategy backtests")
    parser.add_argument("--portfolio-only", action="store_true", help="Only run portfolio strategy backtests")
    parser.add_argument("--export", type=str, default=None, metavar="PATH", help="Export results to JSON file")
    args = parser.parse_args()

    include_trading = not args.portfolio_only
    include_portfolio = not args.trading_only

    results = run_benchmark(
        include_trading=include_trading,
        include_portfolio=include_portfolio,
        quick=args.quick,
        export_path=args.export,
    )

    # Exit code: non-zero if any failures
    sys.exit(1 if results["summary"]["failed"] > 0 else 0)


if __name__ == "__main__":
    main()

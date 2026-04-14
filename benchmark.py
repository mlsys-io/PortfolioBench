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

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# freqtrade lives in a git submodule; add its root so `import freqtrade` resolves.
_FT_ROOT = os.path.join(PROJECT_ROOT, "freqtrade")
if os.path.isdir(os.path.join(_FT_ROOT, "freqtrade")) and _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)

# ---------------------------------------------------------------------------
# Freqtrade imports (deferred to allow graceful failure)
# ---------------------------------------------------------------------------
try:
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode
    from freqtrade.optimize.backtesting import Backtesting
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
    "stocks":  ["AAPL/USD", "MSFT/USD", "NVDA/USD", "GOOG/USD"],
    "indices": ["DJI/USD", "FTSE/USD", "GSPC/USD"],
    "mixed":   ["BTC/USDT", "ETH/USDT", "AAPL/USD", "MSFT/USD", "DJI/USD", "GSPC/USD"],
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
        bt.start()
        bt_results = bt.results

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
    """Pull key metrics from freqtrade backtest result structure.

    Freqtrade backtests return a deeply nested dict. This function searches
    multiple known locations for key performance metrics to ensure we capture
    return, Sharpe, drawdown, win rate, and other details regardless of the
    exact result layout.
    """
    metrics: Dict[str, Any] = {}

    if bt_results is None:
        return metrics

    try:
        # bt.results has structure {"strategy": {name: stats}, "metadata": {...}, ...}
        if isinstance(bt_results, dict) and "strategy" in bt_results:
            strat_dict = bt_results["strategy"]
            if isinstance(strat_dict, dict) and strategy_name in strat_dict:
                strat_result = strat_dict[strategy_name]
            elif isinstance(strat_dict, dict):
                first_key = next(iter(strat_dict), None)
                strat_result = strat_dict.get(first_key, bt_results)
            else:
                strat_result = bt_results
        elif isinstance(bt_results, dict) and strategy_name in bt_results:
            strat_result = bt_results[strategy_name]
        elif isinstance(bt_results, dict):
            first_key = next(iter(bt_results), None)
            strat_result = bt_results.get(first_key, bt_results)
        else:
            return metrics

        if not isinstance(strat_result, dict):
            return metrics

        # Collect all candidate dicts to search for metrics.
        # Freqtrade stores metrics at varying nesting depths depending on version.
        candidates = [strat_result]
        for sub_key in [
            "results_per_pair", "results", "strategy_comparison",
            "backtest_results", "backtest_result",
        ]:
            sub = strat_result.get(sub_key)
            if isinstance(sub, dict):
                candidates.append(sub)
            elif isinstance(sub, list) and sub:
                # results_per_pair is a list of dicts; the TOTAL row is usually last
                for entry in sub:
                    if isinstance(entry, dict):
                        candidates.append(entry)

        # ── Extract trades ──
        for d in candidates:
            for key in ["total_trades", "trade_count"]:
                if key in d:
                    metrics["trades"] = d[key]
                    break
            if "trades" in metrics:
                break
        if "trades" not in metrics:
            trades_df = strat_result.get("trades")
            if trades_df is not None and hasattr(trades_df, "__len__"):
                metrics["trades"] = len(trades_df)

        # ── Extract total return ──
        for d in candidates:
            for key in ["profit_total", "profit_total_pct"]:
                if key in d and d[key] is not None:
                    val = float(d[key])
                    # profit_total is a ratio (0.05 = 5%), profit_total_pct is already %
                    if key == "profit_total":
                        val *= 100
                    metrics["total_return_pct"] = round(val, 2)
                    break
            if "total_return_pct" in metrics:
                break

        # ── Extract absolute profit ──
        for d in candidates:
            if "profit_total_abs" in d and d["profit_total_abs"] is not None:
                metrics["profit_abs"] = round(float(d["profit_total_abs"]), 2)
                break

        # ── Extract max drawdown ──
        for d in candidates:
            for key in ["max_drawdown", "max_drawdown_account", "max_drawdown_abs"]:
                if key in d and d[key] is not None:
                    val = float(d[key])
                    if key in ("max_drawdown", "max_drawdown_account"):
                        val *= 100  # convert ratio to pct
                    metrics["max_drawdown_pct"] = round(val, 2)
                    break
            if "max_drawdown_pct" in metrics:
                break

        # ── Extract Sharpe ratio ──
        for d in candidates:
            for key in ["sharpe", "sharpe_ratio"]:
                if key in d and d[key] is not None:
                    metrics["sharpe"] = round(float(d[key]), 4)
                    break
            if "sharpe" in metrics:
                break

        # ── Extract Sortino ratio ──
        for d in candidates:
            for key in ["sortino", "sortino_ratio"]:
                if key in d and d[key] is not None:
                    metrics["sortino"] = round(float(d[key]), 4)
                    break
            if "sortino" in metrics:
                break

        # ── Extract Calmar ratio ──
        for d in candidates:
            if "calmar" in d and d["calmar"] is not None:
                metrics["calmar"] = round(float(d["calmar"]), 4)
                break

        # ── Extract win rate ──
        for d in candidates:
            for key in ["winrate", "win_rate", "wins", "winning_trades"]:
                if key in d and d[key] is not None:
                    if key in ("winrate", "win_rate"):
                        metrics["win_rate_pct"] = round(float(d[key]) * 100, 2)
                    elif "trades" in metrics and metrics["trades"] > 0:
                        metrics["win_rate_pct"] = round(
                            float(d[key]) / metrics["trades"] * 100, 2
                        )
                    break
            if "win_rate_pct" in metrics:
                break

        # ── Extract profit factor ──
        for d in candidates:
            if "profit_factor" in d and d["profit_factor"] is not None:
                metrics["profit_factor"] = round(float(d["profit_factor"]), 4)
                break

        # ── Extract average trade duration ──
        for d in candidates:
            for key in [
                "holding_avg", "avg_duration", "trade_duration_avg",
                "duration_avg", "holding_avg_s",
            ]:
                if key in d and d[key] is not None:
                    metrics["avg_duration"] = str(d[key])
                    break
            if "avg_duration" in metrics:
                break

        # ── Extract average profit per trade ──
        for d in candidates:
            for key in ["profit_mean", "profit_mean_pct"]:
                if key in d and d[key] is not None:
                    val = float(d[key])
                    if key == "profit_mean":
                        val *= 100
                    metrics["avg_profit_pct"] = round(val, 2)
                    break
            if "avg_profit_pct" in metrics:
                break

        # ── Extract total profit from trades DataFrame when top-level keys missing ──
        if "total_return_pct" not in metrics:
            trades_df = strat_result.get("trades")
            if trades_df is not None and hasattr(trades_df, "profit_abs"):
                try:
                    total_profit = float(trades_df["profit_abs"].sum())
                    wallet = 1_000_000  # default
                    metrics["total_return_pct"] = round(total_profit / wallet * 100, 2)
                    metrics["profit_abs"] = round(total_profit, 2)
                except Exception:
                    pass
            if trades_df is not None and hasattr(trades_df, "profit_ratio"):
                try:
                    metrics.setdefault("trades", len(trades_df))
                    if len(trades_df) > 0:
                        wins = (trades_df["profit_ratio"] > 0).sum()
                        metrics.setdefault(
                            "win_rate_pct", round(float(wins) / len(trades_df) * 100, 2)
                        )
                        avg_pct = float(trades_df["profit_ratio"].mean()) * 100
                        metrics.setdefault("avg_profit_pct", round(avg_pct, 2))
                except Exception:
                    pass

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

        from alpha.BollingerAlpha import BollingerAlpha
        from alpha.MacdAlpha import MacdAlpha
        from alpha.RsiAlpha import RsiAlpha
        from alpha.SimpleEmaFactors import EmaAlpha

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
        from pathlib import Path

        import pandas as pd

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

        # Detect data source: synthetic files are small, real data is larger
        data_source = "unknown"
        try:
            sample = next(data_path.glob("BTC_USDT-1d.feather"), None)
            if sample:
                df_sample = pd.read_feather(sample)
                # Synthetic data has exactly the date range 2024-01-01 to 2026-02-01
                # and very uniform volume; real data typically has more rows
                if len(df_sample) > 800:
                    data_source = "Google Drive (real market data)"
                else:
                    data_source = "Synthetic (generated)"
        except Exception:
            pass

        result["details"] = {
            "total_files": total,
            "valid_files": valid,
            "invalid_files": len(invalid),
            "unique_assets": len(assets),
            "timeframes": sorted(timeframes_found),
            "invalid_list": invalid[:10],  # cap at 10 for readability
            "data_source": data_source,
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
    max_workers: int = 1,
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
    if max_workers > 1:
        print(f"  {_c('Workers:', Colors.DIM)}  {max_workers} parallel processes")
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
            max_workers=max_workers,
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
            max_workers=max_workers,
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
    max_workers: int = 1,
):
    """Run all strategy × asset × timeframe combinations and print results.

    When *max_workers* > 1 the backtests are dispatched to a process pool so
    that independent strategy/asset/timeframe combos execute concurrently.
    """
    # Build the full list of tasks so we can dispatch them all at once.
    tasks: List[Tuple[str, str, str, List[str], str, str]] = []
    for strat in strat_names:
        for asset_label, pairs in asset_configs.items():
            for tf, tf_cfg in tf_configs.items():
                tasks.append((strat, strat_path, asset_label, pairs, tf, tf_cfg["timerange"]))

    # --- parallel execution ------------------------------------------------
    if max_workers > 1 and len(tasks) > 1:
        results_by_key: Dict[Tuple[str, str, str], Dict] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_key = {}
            for strat, sp, asset_label, pairs, tf, timerange in tasks:
                fut = pool.submit(
                    run_single_backtest,
                    strategy_name=strat,
                    strategy_path=sp,
                    pairs=pairs,
                    timeframe=tf,
                    timerange=timerange,
                )
                future_to_key[fut] = (strat, asset_label, tf)

            for fut in as_completed(future_to_key):
                key = future_to_key[fut]
                try:
                    r = fut.result()
                except Exception as exc:
                    r = {
                        "strategy": key[0],
                        "pairs": [],
                        "timeframe": key[2],
                        "timerange": "",
                        "status": "fail",
                        "error": str(exc),
                        "metrics": {},
                        "duration_s": 0.0,
                    }
                r["asset_class"] = key[1]
                results_by_key[key] = r

        # Print results in the original deterministic order.
        for strat, _sp, asset_label, _pairs, tf, _tr in tasks:
            r = results_by_key[(strat, asset_label, tf)]
            result_list.append(r)
    else:
        # --- sequential fallback (max_workers=1) ---------------------------
        for strat, sp, asset_label, pairs, tf, timerange in tasks:
            r = run_single_backtest(
                strategy_name=strat,
                strategy_path=sp,
                pairs=pairs,
                timeframe=tf,
                timerange=timerange,
            )
            r["asset_class"] = asset_label
            result_list.append(r)

    # --- print results & update counters (always in deterministic order) ---
    cur_strat = None
    for r in result_list[len(result_list) - len(tasks):]:
        if r["strategy"] != cur_strat:
            cur_strat = r["strategy"]
            print(subsection(f"Strategy: {cur_strat}"))

        tag = f"{r['asset_class']}/{r['timeframe']}"
        if r["status"] == "pass":
            counters["pass"] += 1
            m = r.get("metrics", {})
            ret_str = format_pct(m.get("total_return_pct"))
            sharpe_str = format_sharpe(m.get("sharpe"))
            trades_str = str(m.get("trades", "?"))
            dd_str = format_pct(m.get("max_drawdown_pct"))
            win_str = format_pct(m.get("win_rate_pct"))
            dur_str = format_duration(r["duration_s"])
            print(detail(
                f"[{status_pass()}] {tag}",
                f"return={ret_str}  sharpe={sharpe_str}  "
                f"DD={dd_str}  win={win_str}  trades={trades_str}  {dur_str}",
            ))
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
            m = r.get("metrics", {})
            strat_best[name] = {
                "return": ret,
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "trades": m.get("trades", 0),
                "asset_class": r.get("asset_class", "?"),
                "timeframe": r.get("timeframe", "?"),
                "drawdown": m.get("max_drawdown_pct"),
                "win_rate": m.get("win_rate_pct"),
                "profit_factor": m.get("profit_factor"),
                "avg_profit": m.get("avg_profit_pct"),
            }

    ranked = sorted(strat_best.items(), key=lambda x: x[1]["return"], reverse=True)

    # Table header
    hdr = (
        f"  {'#':<4}{'Strategy':<30}{'Return':>9}{'Sharpe':>9}{'Sortino':>9}"
        f"{'DD':>9}{'Win%':>8}{'PF':>8}{'Trades':>8}{'Best On':>12}"
    )
    print(_c(hdr, Colors.DIM))
    print(_c("  " + "─" * 102, Colors.DIM))

    for i, (name, info) in enumerate(ranked, 1):
        ret_s = format_pct(info["return"])
        sharpe_s = format_sharpe(info.get("sharpe"))
        sortino_s = format_sharpe(info.get("sortino"))
        trades_s = str(info.get("trades", "?"))
        dd_s = format_pct(info.get("drawdown"))
        win_s = format_pct(info.get("win_rate"))
        pf_s = format_sharpe(info.get("profit_factor"))
        best_s = f"{info['asset_class']}/{info['timeframe']}"
        print(
            f"  {i:<4}{name:<30}{ret_s:>9}{sharpe_s:>9}{sortino_s:>9}"
            f"{dd_s:>9}{win_s:>8}{pf_s:>8}{trades_s:>8}{best_s:>12}"
        )

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
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Number of parallel worker processes for backtests (default: 1 = sequential)",
    )
    args = parser.parse_args()

    include_trading = not args.portfolio_only
    include_portfolio = not args.trading_only

    results = run_benchmark(
        include_trading=include_trading,
        include_portfolio=include_portfolio,
        quick=args.quick,
        export_path=args.export,
        max_workers=max(1, args.workers),
    )

    # Exit code: non-zero if any failures
    sys.exit(1 if results["summary"]["failed"] > 0 else 0)


if __name__ == "__main__":
    main()

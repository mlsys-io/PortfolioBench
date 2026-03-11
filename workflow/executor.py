"""Stage handlers that bridge LumidStack workflow stages to PortfolioBench pipeline.

Each handler has the signature::

    def handler(stage_name: str, params: dict, context: dict) -> dict

The *context* dict is shared across all stages and carries intermediate data
(loaded DataFrames, enriched data, weights, etc.) so that later stages can
consume earlier stages' output — exactly mirroring the seven-step pipeline in
``portfolio/PortfolioManagement.py``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root and freqtrade submodule are importable.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_FT_ROOT = os.path.join(_PROJECT_ROOT, "freqtrade")
if os.path.isdir(os.path.join(_FT_ROOT, "freqtrade")) and _FT_ROOT not in sys.path:
    sys.path.insert(0, _FT_ROOT)

from alpha.SimpleEmaFactors import EmaAlpha  # noqa: E402
from alpha.RsiAlpha import RsiAlpha  # noqa: E402
from alpha.MacdAlpha import MacdAlpha  # noqa: E402
from alpha.BollingerAlpha import BollingerAlpha  # noqa: E402

from portfolio.PortfolioManagement import (  # noqa: E402
    load_pair_data,
    align_close_prices,
    ema_cross_signals,
    build_ema_position_series,
    calculate_ons_weights,
    equal_weight_allocation,
    blend_strategy_weights,
    backtest_portfolio,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# Alpha registry — maps alpha type names to classes.
# ---------------------------------------------------------------------------
ALPHA_REGISTRY: Dict[str, type] = {
    "ema": EmaAlpha,
    "rsi": RsiAlpha,
    "macd": MacdAlpha,
    "bollinger": BollingerAlpha,
}

# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _rsi_signals(df: pd.DataFrame) -> pd.DataFrame:
    """RSI-based entry/exit: enter when oversold, exit when overbought."""
    df = df.copy()
    df["enter_long"] = ((df["rsi_oversold"] == 1) & (df["mean-volume"] > 0.75)).astype(int)
    df["exit_long"] = (df["rsi_overbought"] == 1).astype(int)
    return df


def _macd_signals(df: pd.DataFrame) -> pd.DataFrame:
    """MACD cross entry/exit."""
    df = df.copy()
    cross_above = (df["macd"].shift(1) <= df["macd_signal"].shift(1)) & (
        df["macd"] > df["macd_signal"]
    )
    cross_below = (df["macd"].shift(1) >= df["macd_signal"].shift(1)) & (
        df["macd"] < df["macd_signal"]
    )
    df["enter_long"] = (cross_above & (df["macd_hist_rising"] == 1)).astype(int)
    df["exit_long"] = cross_below.astype(int)
    return df


def _bollinger_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Bands mean-reversion entry/exit."""
    df = df.copy()
    df["enter_long"] = ((df["close"] <= df["bb_lower"]) & (df["mean-volume"] > 0.75)).astype(int)
    df["exit_long"] = (df["close"] >= df["bb_upper"]).astype(int)
    return df


SIGNAL_DISPATCH = {
    "ema_cross": ema_cross_signals,
    "rsi": _rsi_signals,
    "macd": _macd_signals,
    "bollinger": _bollinger_signals,
}


# ============================================================================
# Stage handlers
# ============================================================================

def handle_alpha(stage_name: str, params: dict[str, Any], context: dict[str, Any]) -> dict:
    """Load data and generate alpha indicators.

    Params
    ------
    type : str
        Alpha factor name (``ema``, ``rsi``, ``macd``, ``bollinger``).
        Defaults to ``ema``.
    """
    alpha_type = params.get("type", "ema")
    alpha_cls = ALPHA_REGISTRY.get(alpha_type)
    if alpha_cls is None:
        raise ValueError(
            f"Unknown alpha type {alpha_type!r}. "
            f"Available: {sorted(ALPHA_REGISTRY)}"
        )

    pair_data: Dict[str, pd.DataFrame] = context.get("pair_data", {})
    if not pair_data:
        raise RuntimeError(
            "No pair_data in context. The data loading step must run first. "
            "Ensure backtest.pairs and backtest.data_dir are set in the workflow."
        )

    enriched: Dict[str, pd.DataFrame] = {}
    for pair, df in pair_data.items():
        alpha = alpha_cls(df.copy(), metadata={"pair": pair})
        enriched[pair] = alpha.process()

    context["enriched_data"] = enriched
    context["alpha_type"] = alpha_type
    print(f"[ALPHA] {alpha_type} indicators added for {len(enriched)} pairs")
    return {"alpha_type": alpha_type, "pairs_processed": len(enriched)}


def handle_strategy(stage_name: str, params: dict[str, Any], context: dict[str, Any]) -> dict:
    """Compute entry/exit signals and build position series.

    Params
    ------
    type : str
        Signal generator name.  Defaults to ``ema_cross``.
    volume_threshold : float
        Minimum mean-volume ratio for entry (used by some signal functions).
    """
    strategy_type = params.get("type", "ema_cross")
    signal_fn = SIGNAL_DISPATCH.get(strategy_type)
    if signal_fn is None:
        raise ValueError(
            f"Unknown strategy type {strategy_type!r}. "
            f"Available: {sorted(SIGNAL_DISPATCH)}"
        )

    enriched_data: Dict[str, pd.DataFrame] = context.get("enriched_data", {})
    if not enriched_data:
        raise RuntimeError("No enriched_data in context — alpha stage must run first.")

    ema_positions: Dict[str, pd.Series] = {}
    total_entries = 0
    total_exits = 0

    for pair, df in enriched_data.items():
        df_signals = signal_fn(df)
        pos = build_ema_position_series(df_signals)
        pos.index = df_signals["date"]
        ema_positions[pair] = pos
        entries = int(df_signals["enter_long"].sum())
        exits = int(df_signals["exit_long"].sum())
        total_entries += entries
        total_exits += exits
        print(f"  {pair}: {entries} entries, {exits} exits")

    context["positions"] = ema_positions
    context["strategy_type"] = strategy_type
    print(f"[STRATEGY] {strategy_type} signals computed for {len(ema_positions)} pairs")
    return {
        "strategy_type": strategy_type,
        "total_entries": total_entries,
        "total_exits": total_exits,
    }


def handle_portfolio(stage_name: str, params: dict[str, Any], context: dict[str, Any]) -> dict:
    """Compute portfolio weights via ONS + equal weight blending.

    Params
    ------
    type : str
        Portfolio method (``blend``, ``equal``, ``ons``).  Defaults to ``blend``.
    blend_weights : dict
        ``{"equal": float, "ons": float, "signal": float}`` mixing coefficients
        when *type* is ``blend``.  Default: 0.34 / 0.33 / 0.33.
    ons : dict
        ONS hyper-parameters (``eta``, ``beta``, ``delta``).
    """
    portfolio_type = params.get("type", "blend")

    pair_data: Dict[str, pd.DataFrame] = context.get("pair_data", {})
    positions: Dict[str, pd.Series] = context.get("positions", {})

    prices = align_close_prices(pair_data)
    context["prices"] = prices

    active_pairs = list(prices.columns)
    equal_wt = equal_weight_allocation(active_pairs)

    if portfolio_type == "equal":
        # Pure 1/N allocation — static weights every bar.
        n_bars = len(prices)
        weights = pd.DataFrame(
            {pair: [equal_wt[pair]] * n_bars for pair in active_pairs},
            index=prices.index,
        )
    elif portfolio_type in ("ons", "blend"):
        ons_params = params.get("ons", {})
        ons_weights = calculate_ons_weights(
            prices,
            eta=ons_params.get("eta", 0.0),
            beta=ons_params.get("beta", 1.0),
            delta=ons_params.get("delta", 0.125),
        )

        if portfolio_type == "ons":
            weights = ons_weights
        else:
            # Blended
            bw = params.get("blend_weights", {})
            weights = blend_strategy_weights(
                ons_weights=ons_weights,
                ema_positions=positions,
                equal_wt=equal_wt,
                w_equal=bw.get("equal", 0.34),
                w_ons=bw.get("ons", 0.33),
                w_ema=bw.get("signal", 0.33),
            )
    else:
        raise ValueError(
            f"Unknown portfolio type {portfolio_type!r}. Available: equal, ons, blend"
        )

    context["weights"] = weights
    context["portfolio_type"] = portfolio_type
    print(f"[PORTFOLIO] {portfolio_type} weights computed: {weights.shape}")
    return {"portfolio_type": portfolio_type, "shape": list(weights.shape)}


# ============================================================================
# Public API — register all handlers on a runner
# ============================================================================

def register_all_handlers(runner) -> None:
    """Register the alpha, strategy, and portfolio handlers on *runner*."""
    runner.register_handler("portbench.alpha", handle_alpha)
    runner.register_handler("portbench.strategy", handle_strategy)
    runner.register_handler("portbench.portfolio", handle_portfolio)

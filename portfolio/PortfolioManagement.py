"""
portfolio/main.py
=================
Simple portfolio construction script that combines:
  1. EmaAlpha indicators      (from alpha/SimpleEmaFactors.py)
  2. EmaCross entry/exit signals (from strategy/EmaCrossStrategy.py logic)
  3. ONS (Online Newton Step) weights (from user_data/strategies/ONS.py logic)
  4. Naïve 1/N equal-weight allocation as the baseline

Usage:
    python -m portfolio.PortfolioManagement          # run from project root (PortfolioBench/)
    python portfolio/PortfolioManagement.py          # or directly
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path so we can import alpha/, strategy/, etc.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from alpha.SimpleEmaFactors import EmaAlpha  # reuse existing alpha factor


# ============================================================================
# 1. DATA LOADING
# ============================================================================

def load_pair_data(data_dir: str, pairs: List[str], timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Load feather files for each pair from the data directory.

    Parameters
    ----------
    data_dir  : path to the exchange data folder, e.g. "user_data/data/usstock"
    pairs     : list of pair strings like ["BTC/USDT", "ETH/USDT", ...]
    timeframe : candle timeframe suffix used in filenames (default "1d")

    Returns
    -------
    dict mapping pair name -> OHLCV DataFrame (columns: date, open, high, low, close, volume)
    """
    pair_data = {}
    for pair in pairs:
        # "BTC/USDT" -> "BTC_USDT-1d.feather"
        filename = pair.replace("/", "_") + f"-{timeframe}.feather"
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"[WARN] File not found for {pair}: {filepath} – skipping")
            continue
        df = pd.read_feather(filepath)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values("date").reset_index(drop=True)
        pair_data[pair] = df
        print(f"[DATA] Loaded {pair}: {len(df)} rows, {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    return pair_data


def align_close_prices(pair_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all pairs on 'date' and return a DataFrame of close prices only.
    Columns = pair names, index = date.  Rows with any NaN are dropped.
    """
    frames = []
    for pair, df in pair_data.items():
        s = df.set_index("date")["close"].rename(pair)
        frames.append(s)
    prices = pd.concat(frames, axis=1).dropna()
    print(f"[DATA] Aligned price matrix: {prices.shape[0]} rows × {prices.shape[1]} assets")
    return prices


# ============================================================================
# 2. ALPHA / INDICATOR GENERATION  (reuses alpha/SimpleEmaFactors.py)
# ============================================================================

def generate_alpha_signals(pair_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Run EmaAlpha.process() on every pair's OHLCV DataFrame.
    Adds columns: ema_fast, ema_slow, ema_exit, mean-volume.

    Returns the same dict with enriched DataFrames.
    """
    enriched = {}
    for pair, df in pair_data.items():
        # EmaAlpha expects a DataFrame with OHLCV columns + metadata dict
        alpha = EmaAlpha(df.copy(), metadata={"pair": pair})
        enriched[pair] = alpha.process()
    print(f"[ALPHA] EMA indicators added for {len(enriched)} pairs")
    return enriched


# ============================================================================
# 3. STRATEGY SIGNALS
# ============================================================================

# ---------- 3a. EMA Cross signals (mirrors strategy/EmaCrossStrategy.py) -----

def ema_cross_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entry / exit signals exactly as EmaCrossStrategy does:
      - enter_long = 1  when ema_fast crosses above ema_slow AND mean-volume > 0.75
      - exit_long  = 1  when ema_exit crosses below ema_fast

    Expects df to already contain: ema_fast, ema_slow, ema_exit, mean-volume
    (produced by generate_alpha_signals).
    """
    df = df.copy()

    # Crossed-above: previous bar fast <= slow, current bar fast > slow
    crossed_above = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (df["ema_fast"] > df["ema_slow"])
    df["enter_long"] = ((crossed_above) & (df["mean-volume"] > 0.75)).astype(int)

    # Crossed-below: previous bar exit >= fast, current bar exit < fast
    crossed_below = (df["ema_exit"].shift(1) >= df["ema_fast"].shift(1)) & (df["ema_exit"] < df["ema_fast"])
    df["exit_long"] = crossed_below.astype(int)

    return df


def build_ema_position_series(df: pd.DataFrame) -> pd.Series:
    """
    Convert discrete enter/exit signals into a continuous position flag (0 or 1)
    for each bar.  A position is opened on enter_long and closed on exit_long.
    """
    position = 0
    positions = []
    for _, row in df.iterrows():
        if row["enter_long"] == 1:
            position = 1
        if row["exit_long"] == 1:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=df.index, name="ema_position")


# ---------- 3b. ONS weights (mirrors user_data/strategies/ONS.py) -----------

def calculate_ons_weights(
    prices: pd.DataFrame,
    eta: float = 0.0,
    beta: float = 1.0,
    delta: float = 0.125,
) -> pd.DataFrame:
    """
    Online Newton Step portfolio optimisation.
    Reproduces the logic in ONS_Portfolio.calculate_ons_weights().

    Parameters
    ----------
    prices : DataFrame of aligned close prices (columns = pairs)
    eta    : mixing parameter toward uniform (0 = pure ONS)
    beta   : controls gradient scaling
    delta  : step-size scaling for the quadratic update

    Returns
    -------
    DataFrame of portfolio weights per bar (columns = pairs, rows = dates)
    """
    pairs = list(prices.columns)
    n_assets = len(pairs)
    n_rows = len(prices)

    A = np.eye(n_assets)
    b = np.zeros(n_assets)
    p = np.ones(n_assets) / n_assets  # start equal-weight

    weights_history = np.zeros((n_rows, n_assets))
    price_vals = prices.values

    # Price-relative vectors: r_t = price_t / price_{t-1}  (first row = 1)
    r_matrix = np.vstack([np.ones(n_assets), price_vals[1:] / price_vals[:-1]])

    for t in range(n_rows):
        weights_history[t] = p
        r_t = r_matrix[t]

        port_ret = max(np.dot(p, r_t), 1e-8)
        grad = r_t / port_ret

        A += np.outer(grad, grad)                   # volatility adjustment
        b += (1 + 1.0 / beta) * grad                # profit adjustment

        A_inv = np.linalg.pinv(A)
        q = delta * A_inv.dot(b)

        # Project onto probability simplex under A-norm
        p_next = _project_simplex_A_norm(q, A, n_assets)

        if eta > 0:
            uniform = np.ones(n_assets) / n_assets
            p_next = (1 - eta) * p_next + eta * uniform

        p = p_next

    weights_df = pd.DataFrame(weights_history, index=prices.index, columns=pairs)
    print(f"[ONS] Weights computed: {n_rows} bars × {n_assets} assets")
    return weights_df


def _project_simplex_A_norm(q: np.ndarray, A: np.ndarray, n: int) -> np.ndarray:
    """Solve for the A-norm projection onto the simplex (sum = 1.0)."""
    def objective(p):
        diff = q - p
        return diff.T @ A @ diff

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
    bounds = tuple((0.0, 1.0) for _ in range(n))
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, tol=1e-6)
    return res.x


# ============================================================================
# 4. PORTFOLIO ALLOCATION  (1/N + strategy blending)
# ============================================================================

def equal_weight_allocation(pairs: List[str]) -> Dict[str, float]:
    """
    Simplest possible allocation: 1/N for each asset.
    """
    n = len(pairs)
    weights = {pair: 1.0 / n for pair in pairs}
    return weights


def blend_strategy_weights(
    ons_weights: pd.DataFrame,
    ema_positions: Dict[str, pd.Series],
    equal_wt: Dict[str, float],
    w_equal: float = 0.34,
    w_ons: float = 0.33,
    w_ema: float = 0.33,
) -> pd.DataFrame:
    """
    Combine three weight sources into a single blended portfolio.

    Strategy weights:
      - equal_wt      : static 1/N across all pairs
      - ons_weights    : dynamic ONS-based weights (DataFrame, rows=dates)
      - ema_positions  : binary 0/1 per pair (Dict of Series aligned to same index)

    Blending formula per bar:
      final_w[pair] = w_equal * (1/N)
                    + w_ons   * ons_weight[pair]
                    + w_ema   * ema_position[pair] * (1/N)   # EMA "turns on/off" its 1/N slice

    After blending, weights are re-normalised to sum to 1.

    Parameters
    ----------
    w_equal : blend weight for equal-weight component   (default ~1/3)
    w_ons   : blend weight for ONS component            (default ~1/3)
    w_ema   : blend weight for EMA cross component      (default ~1/3)

    Returns
    -------
    DataFrame of final portfolio weights (columns = pairs, rows = dates)
    """
    pairs = list(ons_weights.columns)
    index = ons_weights.index
    n = len(pairs)

    blended = pd.DataFrame(index=index, columns=pairs, dtype=float)

    for pair in pairs:
        eq = equal_wt[pair]                                # constant scalar
        ons_col = ons_weights[pair].values                 # array, one per bar
        ema_col = ema_positions[pair].reindex(index).fillna(0).values  # 0 or 1

        # Combine: each component contributes its weighted share
        blended[pair] = w_equal * eq + w_ons * ons_col + w_ema * ema_col * (1.0 / n)

    # Re-normalise each row so weights sum to 1 (avoid division by zero)
    row_sums = blended.sum(axis=1).replace(0, 1)
    blended = blended.div(row_sums, axis=0)

    print(f"[BLEND] Final weights computed with mix equal={w_equal}, ons={w_ons}, ema={w_ema}")
    return blended


# ===========================================================================
# 5. PORTFOLIO BACKTEST (simple daily return tracking)
# ===========================================================================

def backtest_portfolio(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """
    Walk-forward backtest: each bar, portfolio return = sum(weight_i * return_i).

    Parameters
    ----------
    prices          : aligned close price DataFrame (columns = pairs)
    weights         : portfolio weight DataFrame (same shape/index as prices)
    initial_capital : starting dollar value

    Returns
    -------
    DataFrame with columns: date, portfolio_value, daily_return
    """
    # Daily simple returns for each asset
    returns = prices.pct_change().fillna(0)

    # Portfolio return per bar = weighted sum of individual returns
    port_returns = (weights.shift(1) * returns).sum(axis=1)  # shift: use yesterday's weight for today's return

    # Cumulative portfolio value
    cum_value = initial_capital * (1 + port_returns).cumprod()

    result = pd.DataFrame({
        "date": prices.index,
        "portfolio_value": cum_value.values,
        "daily_return": port_returns.values,
    })
    return result


# ============================================================================
# 6. PERFORMANCE METRICS
# ============================================================================

def compute_metrics(result: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic portfolio performance stats.
    """
    rets = result["daily_return"]
    total_return = result["portfolio_value"].iloc[-1] / result["portfolio_value"].iloc[0] - 1

    n_bars = len(rets)
    ann_factor = 365  # crypto trades every day
    mean_daily = rets.mean()
    std_daily = rets.std()

    sharpe = (mean_daily / std_daily) * np.sqrt(ann_factor) if std_daily > 0 else 0.0

    # Maximum drawdown
    cum_max = result["portfolio_value"].cummax()
    drawdown = (result["portfolio_value"] - cum_max) / cum_max
    max_dd = drawdown.min()

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "annualised_return_pct": round(((1 + total_return) ** (ann_factor / max(n_bars, 1)) - 1) * 100, 2),
        "annualised_sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "n_bars": n_bars,
    }
    return metrics


# ============================================================================
# 7. MAIN – ties everything together
# ============================================================================

def run_portfolio(
    data_dir: str = None,
    pairs: List[str] = None,
    timeframe: str = "1d",
    initial_capital: float = 10_000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    End-to-end pipeline:
      1. Load data
      2. Generate EMA alpha indicators
      3. Compute EMA cross entry/exit signals & positions
      4. Compute ONS weights
      5. Compute 1/N equal weights
      6. Blend all three into final portfolio weights
      7. Backtest & report metrics

    Returns
    -------
    (backtest_result, final_weights, metrics)
    """
    # -- defaults --
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "user_data", "data", "usstock")
    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "MSFT/USD"]

    # Step 1: Load raw OHLCV data
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    pair_data = load_pair_data(data_dir, pairs, timeframe)
    if len(pair_data) == 0:
        raise RuntimeError("No data loaded – check your data directory and pair names.")

    # Step 2: Generate alpha indicators (EMA fast/slow/exit + mean-volume)
    print("\n" + "=" * 60)
    print("STEP 2 — Generating EMA alpha indicators")
    print("=" * 60)
    enriched_data = generate_alpha_signals(pair_data)

    # Step 3: Compute EMA cross entry/exit signals → binary position per pair
    print("\n" + "=" * 60)
    print("STEP 3 — Computing EMA Cross strategy signals")
    print("=" * 60)
    ema_positions = {}
    for pair, df in enriched_data.items():
        df_signals = ema_cross_signals(df)
        pos = build_ema_position_series(df_signals)
        # Align position series to date index for merging later
        pos.index = df_signals["date"]
        ema_positions[pair] = pos
        entry_count = df_signals["enter_long"].sum()
        exit_count = df_signals["exit_long"].sum()
        print(f"  {pair}: {entry_count} entries, {exit_count} exits")

    # Step 4: Align close prices and compute ONS weights
    print("\n" + "=" * 60)
    print("STEP 4 — Computing ONS (Online Newton Step) weights")
    print("=" * 60)
    prices = align_close_prices(pair_data)
    ons_weights = calculate_ons_weights(prices, eta=0.0, beta=1.0, delta=0.125)

    # Step 5: Compute static 1/N equal weights
    print("\n" + "=" * 60)
    print("STEP 5 — Setting up 1/N equal-weight allocation")
    print("=" * 60)
    active_pairs = list(prices.columns)
    equal_wt = equal_weight_allocation(active_pairs)
    print(f"  Equal weight per asset: {equal_wt[active_pairs[0]]:.4f} ({len(active_pairs)} assets)")

    # Step 6: Blend all strategy weights
    print("\n" + "=" * 60)
    print("STEP 6 — Blending strategies into final portfolio")
    print("=" * 60)
    final_weights = blend_strategy_weights(
        ons_weights=ons_weights,
        ema_positions=ema_positions,
        equal_wt=equal_wt,
        w_equal=0.34,   # ~1/3 to passive equal-weight
        w_ons=0.33,     # ~1/3 to ONS adaptive weights
        w_ema=0.33,     # ~1/3 to EMA-cross momentum overlay
    )
    print("  Sample final weights (last bar):")
    last_row = final_weights.iloc[-1]
    for pair in active_pairs:
        print(f"    {pair}: {last_row[pair]:.4f}")

    # Step 7: Backtest the blended portfolio
    print("\n" + "=" * 60)
    print("STEP 7 — Running backtest")
    print("=" * 60)
    result = backtest_portfolio(prices, final_weights, initial_capital)
    metrics = compute_metrics(result)

    print(f"\n{'=' * 60}")
    print("PORTFOLIO RESULTS")
    print(f"{'=' * 60}")
    print(f"  Initial capital      : ${initial_capital:,.2f}")
    print(f"  Final value          : ${result['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  Total return         : {metrics['total_return_pct']:.2f}%")
    print(f"  Annualised return    : {metrics['annualised_return_pct']:.2f}%")
    print(f"  Annualised Sharpe    : {metrics['annualised_sharpe']:.4f}")
    print(f"  Max drawdown         : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Number of bars       : {metrics['n_bars']}")

    return result, final_weights, metrics


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    result, weights, metrics = run_portfolio()
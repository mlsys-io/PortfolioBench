"""Feature engineering for the direct event-probability model.

All features are computed from information strictly available at observation
time t.  No future data relative to t is used anywhere in this module.

On-chain metrics are shifted 24 hours before use to guard against same-day
forward-fill look-ahead (``market-cap`` is excluded entirely because it is
derived directly from BTC price).

Feature groups
--------------
Contract-level (4):  log_moneyness, log_h_remaining, sigma_h, bs_prob
BTC price (8):       log_ret_1h, log_ret_24h, log_ret_7d,
                     realized_vol_24h, realized_vol_7d, vol_ratio,
                     rsi_14, momentum_7d
On-chain lagged (3): mvrv_lag24, hash-rate_lag24, difficulty_lag24
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Feature column name lists — canonical order must match between training
# and inference.
# ---------------------------------------------------------------------------

CONTRACT_FEATURE_COLS: list[str] = [
    "log_moneyness",
    "log_h_remaining",
    "sigma_h",
    "bs_prob",
]

BTC_FEATURE_COLS: list[str] = [
    "log_ret_1h",
    "log_ret_24h",
    "log_ret_7d",
    "realized_vol_24h",
    "realized_vol_7d",
    "vol_ratio",
    "rsi_14",
    "momentum_7d",
]

ONCHAIN_FEATURE_COLS: list[str] = [
    "mvrv_lag24",
    "hash-rate_lag24",
    "difficulty_lag24",
]

ALL_FEATURE_COLS: list[str] = (
    CONTRACT_FEATURE_COLS + BTC_FEATURE_COLS + ONCHAIN_FEATURE_COLS
)


# ---------------------------------------------------------------------------
# BTC-level feature engineering
# ---------------------------------------------------------------------------

def add_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC-level and on-chain features.

    Each feature at row ``i`` uses only rows ``0..i`` (strictly causal).
    On-chain columns (``mvrv``, ``hash-rate``, ``difficulty``) are optional;
    missing columns produce NaN feature columns.

    Args:
        df: DataFrame from :func:`~polymarket.settlement.load_btc_hourly`.
            Must have columns: ``dt`` (UTC-aware), ``Close``.

    Returns:
        Copy of ``df`` with new feature columns appended.
    """
    df = df.copy()

    # ---- Price returns ----
    df["log_ret_1h"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_ret_24h"] = np.log(df["Close"] / df["Close"].shift(24))
    df["log_ret_7d"] = np.log(df["Close"] / df["Close"].shift(168))

    # ---- Realized volatility ----
    df["realized_vol_24h"] = df["log_ret_1h"].rolling(24, min_periods=12).std()
    df["realized_vol_7d"] = df["log_ret_1h"].rolling(168, min_periods=48).std()
    df["vol_ratio"] = df["realized_vol_24h"] / df["realized_vol_7d"].replace(0.0, np.nan)

    # ---- RSI-14 (no TA-Lib dependency) ----
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=7).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=7).mean()
    rs = gain / loss.replace(0.0, np.nan)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ---- Momentum ----
    df["momentum_7d"] = df["Close"] / df["Close"].shift(168).replace(0.0, np.nan) - 1.0

    # ---- On-chain features (24h lag; market-cap excluded) ----
    for src_col, feat_col in [
        ("mvrv",        "mvrv_lag24"),
        ("hash-rate",   "hash-rate_lag24"),
        ("difficulty",  "difficulty_lag24"),
    ]:
        df[feat_col] = df[src_col].shift(24) if src_col in df.columns else np.nan

    return df


# ---------------------------------------------------------------------------
# Contract-specific feature engineering
# ---------------------------------------------------------------------------

def add_contract_features(
    df: pd.DataFrame,
    K: float,
    T: pd.Timestamp,
) -> pd.DataFrame:
    """Add contract-specific features for a given (K, T) pair.

    Must be called after :func:`add_btc_features` since it relies on
    ``realized_vol_7d``.

    Args:
        df: DataFrame with BTC features and a ``dt`` column (UTC-aware).
        K:  Strike price in USD.
        T:  Settlement timestamp (UTC-aware Timestamp).

    Returns:
        Copy of ``df`` with contract feature columns added.
    """
    df = df.copy()
    T = T if T.tzinfo is not None else T.tz_localize("UTC")

    # ---- Moneyness: log(current_price / strike) ----
    # Positive = already above strike; negative = below strike.
    df["log_moneyness"] = np.log(df["Close"] / K)

    # ---- Hours remaining until T ----
    h_remaining = (T - df["dt"]).dt.total_seconds() / 3600.0
    h_remaining = h_remaining.clip(lower=0.5)   # floor at 30 min to avoid log(0)
    df["h_remaining"] = h_remaining
    df["log_h_remaining"] = np.log(h_remaining)

    # ---- Scaled volatility: sigma_1h * sqrt(h) ----
    sigma_1h = df["realized_vol_7d"].clip(lower=1e-6)
    df["sigma_h"] = sigma_1h * np.sqrt(h_remaining)

    # ---- Black-Scholes naive probability (model baseline) ----
    # P(BTC_T > K) under driftless lognormal using current sigma estimate.
    log_K_over_S = -df["log_moneyness"]          # = log(K / Close)
    sigma_h_safe = df["sigma_h"].replace(0.0, np.nan)
    df["bs_prob"] = df.apply(
        lambda row: float(norm.sf(
            -row["log_moneyness"] / row["sigma_h"]
        )) if row["sigma_h"] > 0 else float("nan"),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    K: float,
    T: pd.Timestamp,
) -> pd.DataFrame:
    """Run the full feature pipeline on a BTC DataFrame.

    Returns a copy of ``df`` enriched with all feature columns plus
    ``h_remaining``.  Rows with any NaN or Inf in :data:`ALL_FEATURE_COLS`
    are dropped.

    Callers use ``result[ALL_FEATURE_COLS]`` for the model input matrix and
    ``result["dt"]`` for index alignment.

    Args:
        df: BTC DataFrame from :func:`~polymarket.settlement.load_btc_hourly`.
        K:  Contract strike price.
        T:  Contract settlement timestamp (UTC).

    Returns:
        Filtered DataFrame retaining only rows with complete features.
    """
    df = add_btc_features(df)
    df = add_contract_features(df, K, T)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ALL_FEATURE_COLS)
    return df

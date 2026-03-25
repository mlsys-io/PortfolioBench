"""Construct synthetic training samples for the direct event-probability model.

Training data is generated entirely from ``data_1h.csv`` without any
real Polymarket contract data.  For each synthetic event ``(K, T)``:

* ``T`` is a weekly settlement time (default: Monday 17:00 UTC).
* ``K`` is a strike derived by scaling the BTC price ``window_days`` before
  ``T`` by a set of relative multipliers.
* Samples span every hourly bar in ``[T - window_days * 24h, T)``.
* Label ``y = 1`` if ``BTC_Open_at_T > K`` else ``0``.

This framing mirrors real Polymarket "Will BTC be above $K on date T?" contracts
and produces a large, temporally ordered dataset spanning years of BTC history.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from polymarket.event_features import (
    ALL_FEATURE_COLS,
    add_btc_features,
    add_contract_features,
)
from polymarket.settlement import get_resolution_price

logger = logging.getLogger(__name__)

# Default relative strikes: 15% below to 15% above in 5% increments.
DEFAULT_RELATIVE_STRIKES: list[float] = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weekly_settlement_times(
    btc_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    day_of_week: int = 0,
    hour_utc: int = 17,
) -> list[pd.Timestamp]:
    """Return all candidate settlement timestamps within [start_date, end_date).

    Only timestamps that have a matching hourly candle in ``btc_df`` are kept.
    """
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")

    dt_range = pd.date_range(start=start_ts, end=end_ts, freq="h", tz="UTC")
    candidates = dt_range[
        (dt_range.dayofweek == day_of_week) & (dt_range.hour == hour_utc)
    ]

    btc_ts_set: set = set(btc_df["dt"].values)
    return [ts for ts in candidates if ts.to_datetime64() in btc_ts_set]


def _reference_price(btc_df: pd.DataFrame, target_ts: pd.Timestamp) -> Optional[float]:
    """Return Close price at ``target_ts``, or the nearest bar within ±2h."""
    row = btc_df[btc_df["dt"] == target_ts]
    if not row.empty:
        return float(row["Close"].iloc[0])

    nearby = btc_df[
        (btc_df["dt"] >= target_ts - pd.Timedelta(hours=2))
        & (btc_df["dt"] <= target_ts + pd.Timedelta(hours=2))
    ]
    return float(nearby.iloc[0]["Close"]) if not nearby.empty else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_samples(
    btc_df: pd.DataFrame,
    start_date: str = "2018-01-01",
    end_date: str = "2025-06-01",
    window_days: int = 7,
    day_of_week: int = 0,
    hour_utc: int = 17,
    relative_strikes: Optional[list[float]] = None,
    strike_round_to: int = 1000,
) -> pd.DataFrame:
    """Build a labelled training dataset from synthetic weekly BTC events.

    For each settlement time ``T``, strikes ``K`` are set from the BTC price
    ``window_days`` before ``T``.  Every hourly bar in the window
    ``[T - window_days * 24h, T)`` becomes a training sample with label
    ``y = 1 if BTC_Open_at_T > K else 0``.

    Args:
        btc_df:           DataFrame from :func:`~polymarket.settlement.load_btc_hourly`.
        start_date:       Earliest settlement date to include (ISO string).
        end_date:         Latest settlement date to include, exclusive (ISO string).
        window_days:      Contract duration in days (default 7).
        day_of_week:      Weekday for settlement: 0=Mon … 6=Sun (default 0=Mon).
        hour_utc:         Hour of settlement in UTC (default 17).
        relative_strikes: Multipliers applied to reference BTC price to set K.
                          Default: ``[0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]``.
        strike_round_to:  Round strikes to nearest N USD (default 1000).

    Returns:
        DataFrame with columns ``[*ALL_FEATURE_COLS, 'dt', 'h_remaining',
        'label', 'K', 'T']``.  Each row is one training sample.
        ``T`` is a UTC-aware Timestamp used for temporal train/val splitting.

    Raises:
        ValueError: If no samples can be generated.
    """
    if relative_strikes is None:
        relative_strikes = DEFAULT_RELATIVE_STRIKES

    settlement_times = _weekly_settlement_times(
        btc_df, start_date, end_date, day_of_week, hour_utc
    )
    logger.info(
        "Building training data: %d settlement times × %d strikes × ~%dh window",
        len(settlement_times),
        len(relative_strikes),
        window_days * 24,
    )

    # Pre-compute BTC features once for the full series (strictly causal rolling
    # windows use only past rows, so computing on the full series is safe).
    btc_feats_df = add_btc_features(btc_df)

    all_chunks: list[pd.DataFrame] = []

    for T in settlement_times:
        # ---- Resolution price (label source) ----
        try:
            resolution_price = get_resolution_price(btc_df, T.isoformat())
        except ValueError:
            logger.debug("Skipping T=%s: no settlement candle", T)
            continue

        # ---- Reference price for strike setting (BTC price window_days before T) ----
        ref_ts = T - pd.Timedelta(days=window_days)
        ref_price = _reference_price(btc_df, ref_ts)
        if ref_price is None:
            logger.debug("Skipping T=%s: no reference price for strikes", T)
            continue

        # ---- Strike set for this event ----
        strikes = sorted({
            int(round(ref_price * r / strike_round_to) * strike_round_to)
            for r in relative_strikes
        })
        strikes = [k for k in strikes if k > 0]

        # ---- Feature window [T - window_days, T) ----
        T_start = T - pd.Timedelta(days=window_days)
        window_base = btc_feats_df[
            (btc_feats_df["dt"] >= T_start) & (btc_feats_df["dt"] < T)
        ].copy()

        if window_base.empty:
            continue

        for K in strikes:
            label = int(resolution_price > K)

            contract_df = add_contract_features(window_base, float(K), T)
            contract_df = contract_df.replace([np.inf, -np.inf], np.nan)
            contract_df = contract_df.dropna(subset=ALL_FEATURE_COLS)

            if contract_df.empty:
                continue

            chunk = contract_df[ALL_FEATURE_COLS + ["dt", "h_remaining"]].copy()
            chunk["label"] = label
            chunk["K"] = float(K)
            chunk["T"] = T
            all_chunks.append(chunk)

    if not all_chunks:
        raise ValueError(
            "No training samples were generated.  "
            f"Check that btc_df covers [{start_date}, {end_date}] "
            "and that settlement times fall on weekdays with BTC data."
        )

    result = pd.concat(all_chunks, ignore_index=True)
    logger.info(
        "Generated %d samples from %d events (%d events had data)",
        len(result),
        len(settlement_times),
        len(all_chunks) // max(1, len(relative_strikes)),
    )
    return result

"""Derive binary contract settlement from historical BTC OHLCV data.

Resolution rule (from Polymarket contract description)
------------------------------------------------------
"This market will resolve to 'Yes' if the Binance 1 minute candle for
BTC/USDT **12:00 in the ET timezone (noon)** on the date specified in the
title has a final 'Close' price higher than the price specified."

Implementation
--------------
* Noon ET = 12:00 PM US/Eastern.  In January (EST) that is UTC−5, so noon
  ET = **17:00:00 UTC**.
* The Polymarket ``endDate`` field stores this as ``"2026-01-20T17:00:00Z"``.
* In the hourly OHLCV data (``data_1h.csv`` / feather), each row is
  timestamped at the **start** of the hour.  The row at ``17:00 UTC``
  covers the period ``17:00–18:00 UTC``.
* The 1-minute candle close at ``17:00:00`` is best approximated by the
  **Open** of that hourly row (= price at the instant the hour begins).
* Verified against Jan-20 contracts: Open=90,064 at 17:00 UTC explains
  $90k YES resolving YES and $92k YES resolving NO.
"""

from __future__ import annotations

import pandas as pd


def load_btc_hourly(csv_path: str) -> pd.DataFrame:
    """Load ``data_1h.csv`` and return a DataFrame with a UTC-aware ``dt``
    column and standard OHLCV columns.

    The CSV uses Unix epoch seconds in the ``Timestamp`` column.

    Args:
        csv_path: Path to the hourly BTC CSV file.

    Returns:
        DataFrame with columns [dt, Open, High, Low, Close, Volume, ...].
        ``dt`` is timezone-aware (UTC).
    """
    df = pd.read_csv(csv_path)
    df["dt"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    return df


def get_resolution_price(btc_df: pd.DataFrame, end_date_utc: str) -> float:
    """Return the BTC price used to settle a Polymarket contract.

    The resolution price is the **Open** of the hourly candle whose timestamp
    equals ``end_date_utc``.  See module docstring for the derivation.

    Args:
        btc_df:       DataFrame as returned by :func:`load_btc_hourly`.
        end_date_utc: Contract ``endDate`` field, e.g. ``"2026-01-20T17:00:00Z"``.

    Returns:
        float: BTC Open price at the resolution candle.

    Raises:
        ValueError: If no candle is found at the given timestamp.
    """
    ts = pd.Timestamp(end_date_utc).tz_localize("UTC") if pd.Timestamp(end_date_utc).tzinfo is None \
        else pd.Timestamp(end_date_utc)

    row = btc_df[btc_df["dt"] == ts]
    if row.empty:
        raise ValueError(
            f"No hourly candle found at {ts}.  "
            "Check that data_1h.csv covers the contract expiry date."
        )
    return float(row["Open"].iloc[0])


def compute_settlement(resolution_price: float, strike: float, direction: str = "above") -> float:
    """Return 1.0 (YES wins) or 0.0 (NO wins) for a binary contract.

    Args:
        resolution_price: BTC price at contract expiry (from :func:`get_resolution_price`).
        strike:           Contract strike price in USD.
        direction:        ``'above'`` (price > strike → YES) or
                          ``'below'`` (price < strike → YES).

    Returns:
        1.0 if YES resolves, 0.0 if NO resolves.
    """
    if direction == "above":
        return 1.0 if resolution_price > strike else 0.0
    if direction == "below":
        return 1.0 if resolution_price < strike else 0.0
    raise ValueError(f"Unknown direction: {direction!r}.  Expected 'above' or 'below'.")


def verify_settlements(
    contracts: list,
    btc_df: pd.DataFrame,
) -> list[dict]:
    """Cross-check :attr:`ContractMetadata.settlement` against BTC price data.

    This is a diagnostic helper.  It recomputes settlements from BTC data and
    compares against the ``outcomePrices``-derived value stored on each contract.

    Args:
        contracts: List of :class:`~polymarket.contracts.ContractMetadata`.
        btc_df:    DataFrame as returned by :func:`load_btc_hourly`.

    Returns:
        List of dicts with keys: slug, strike, direction, resolution_price,
        outcome_prices_settlement, btc_derived_settlement, match.
    """
    results = []
    for c in contracts:
        try:
            res_price = get_resolution_price(btc_df, c.end_date_utc)
            btc_settlement = compute_settlement(res_price, c.strike, c.direction)
            match = btc_settlement == c.settlement
        except ValueError as exc:
            res_price = float("nan")
            btc_settlement = float("nan")
            match = False
            _ = exc  # logged via return value

        results.append(
            {
                "slug": c.slug,
                "strike": c.strike,
                "direction": c.direction,
                "resolution_price": res_price,
                "outcome_prices_settlement": c.settlement,
                "btc_derived_settlement": btc_settlement,
                "match": match,
            }
        )
    return results

"""Generate synthetic hourly OHLCV price series for Polymarket contracts.

Why synthetic prices?
---------------------
Real Polymarket hourly OHLCV is not currently available.  We approximate the
contract price path using the **risk-neutral log-normal model**:

    p(t) = P(BTC_T > K | BTC_t) = norm.sf(log(K/BTC_t), loc=0, scale=σ_h·√h)

where:

* ``BTC_t``  — observed hourly BTC close from ``data_1h.csv``.
* ``σ_h``    — historical 1-hour BTC log-return std dev (calibration window
               = 6 months before contract start).
* ``h``      — hours remaining until expiry at time ``t``.
* ``mu = 0`` — risk-neutral assumption.  Using the model's ``mu`` would embed
               the model's directional edge into the synthetic price, making
               the backtest circular.

A small Gaussian noise term (σ_noise = 0.015 by default) is added per candle
to prevent a perfectly smooth theoretical curve and to mimic market noise.

Synthetic OHLCV construction
-----------------------------
* ``close`` = ``p_market(t)``
* ``open``  = previous candle's ``close`` (first candle: same as close)
* ``high``  = max(open, close) + half the candle noise (simulated intracandle)
* ``low``   = min(open, close) − half the candle noise
* ``volume``= 1.0  (placeholder; not used by the strategy)
* Final candle close is **patched to the settlement value** (0.001 or 0.999)
  so that P&L is computed correctly.

Limitations
-----------
* No adverse selection, bid-ask spread, or liquidity premium.
* Noise is i.i.d. Gaussian; real microstructure exhibits autocorrelation.
* The synthetic price appears better calibrated than actual market prices
  because it is generated from the same distributional assumption the model
  uses.  Treat backtest results as a consistency check, not a performance
  estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polymarket.contracts import ContractMetadata

# Polymarket price bounds (matches exchange/polymarket.py)
PRICE_FLOOR = 0.001
PRICE_CEIL = 0.999


def _calibrate_sigma(
    btc_df: pd.DataFrame,
    before_ts: pd.Timestamp,
    months: int = 6,
) -> float:
    """Compute the historical 1-hour log-return std dev from BTC data.

    Uses the ``months`` calendar months immediately before ``before_ts`` as the
    calibration window.

    Args:
        btc_df:    DataFrame with columns ``dt`` (UTC-aware) and ``Close``.
        before_ts: Upper bound (exclusive) of the calibration window.
        months:    Number of calendar months to look back.

    Returns:
        float: Annualised-adjusted 1-hour σ.  Raises ``ValueError`` if there
        is insufficient data.
    """
    cutoff = before_ts - pd.DateOffset(months=months)
    mask = (btc_df["dt"] >= cutoff) & (btc_df["dt"] < before_ts)
    subset = btc_df[mask]["Close"]
    if len(subset) < 100:
        raise ValueError(
            f"Insufficient BTC data for sigma calibration: found {len(subset)} rows "
            f"in window [{cutoff}, {before_ts})."
        )
    log_rets = np.log(subset / subset.shift(1)).dropna()
    return float(log_rets.std())


def build_synthetic_ohlcv(
    btc_df: pd.DataFrame,
    contract: ContractMetadata,
    sigma_1h: float | None = None,
    noise_std: float = 0.015,
    random_seed: int = 42,
    calibration_months: int = 6,
) -> pd.DataFrame:
    """Build a synthetic hourly OHLCV DataFrame for one contract.

    The returned DataFrame uses the same column names and dtypes as the feather
    files expected by PortfolioBench's ``FeatherDataHandler``:
    ``date`` (millisecond int), ``open``, ``high``, ``low``, ``close``,
    ``volume``.

    Args:
        btc_df:             BTC hourly DataFrame from
                            :func:`~polymarket.settlement.load_btc_hourly`.
        contract:           :class:`~polymarket.contracts.ContractMetadata` instance.
        sigma_1h:           Pre-computed 1-hour BTC σ.  If ``None``, calibrated
                            automatically from ``btc_df``.
        noise_std:          Std dev of per-candle Gaussian market noise added to
                            the theoretical fair value.
        random_seed:        Seed for reproducibility.
        calibration_months: Months of history used if ``sigma_1h`` is ``None``.

    Returns:
        DataFrame with columns [date, open, high, low, close, volume].
        ``date`` values are UTC millisecond integers (freqtrade convention).
    """
    rng = np.random.default_rng(random_seed)

    start_ts = pd.Timestamp(contract.start_date_utc).tz_localize("UTC") \
        if pd.Timestamp(contract.start_date_utc).tzinfo is None \
        else pd.Timestamp(contract.start_date_utc)
    end_ts = pd.Timestamp(contract.end_date_utc).tz_localize("UTC") \
        if pd.Timestamp(contract.end_date_utc).tzinfo is None \
        else pd.Timestamp(contract.end_date_utc)

    # Align start to the nearest hour boundary on or after start_date
    start_ts = start_ts.ceil("h")

    # Calibrate sigma if not supplied
    if sigma_1h is None:
        sigma_1h = _calibrate_sigma(btc_df, start_ts, months=calibration_months)

    # Select BTC candles covering the contract lifetime [start_ts, end_ts]
    mask = (btc_df["dt"] >= start_ts) & (btc_df["dt"] <= end_ts)
    btc_window = btc_df[mask].copy().reset_index(drop=True)

    if btc_window.empty:
        raise ValueError(
            f"No BTC candles found between {start_ts} and {end_ts}.  "
            "Check data_1h.csv coverage."
        )

    n = len(btc_window)
    closes = btc_window["Close"].values.astype(float)
    timestamps = btc_window["dt"].values  # numpy datetime64[ns, UTC]

    # For each candle compute fair value and add noise
    fair_values = np.empty(n)
    for i in range(n):
        # Hours remaining until expiry (end_ts is the last candle)
        h_remaining = n - 1 - i  # 0 at the last candle
        btc_t = closes[i]

        if h_remaining <= 0:
            # Settlement candle: clamp to PRICE_FLOOR / PRICE_CEIL
            fair_values[i] = PRICE_CEIL if contract.settlement == 1.0 else PRICE_FLOOR
            continue

        log_dist = np.log(contract.strike / btc_t)
        sigma_h = sigma_1h * np.sqrt(h_remaining)

        # Risk-neutral P(BTC_T > K): use norm.sf with loc=0
        from scipy.stats import norm  # local import to keep module lightweight
        p_fair = float(norm.sf(log_dist, loc=0.0, scale=sigma_h))

        noise = rng.normal(0.0, noise_std)
        p_market = float(np.clip(p_fair + noise, PRICE_FLOOR, PRICE_CEIL))

        # For 'below' direction, YES = P(BTC < K) = 1 - P(BTC > K)
        if contract.direction == "below":
            p_market = 1.0 - p_market

        fair_values[i] = p_market

    # Build OHLCV arrays
    open_arr = np.empty(n)
    open_arr[0] = fair_values[0]
    open_arr[1:] = fair_values[:-1]

    intracandle_noise = np.abs(rng.normal(0.0, noise_std / 2.0, size=n))
    high_arr = np.clip(np.maximum(open_arr, fair_values) + intracandle_noise, PRICE_FLOOR, PRICE_CEIL)
    low_arr = np.clip(np.minimum(open_arr, fair_values) - intracandle_noise, PRICE_FLOOR, PRICE_CEIL)

    # Convert timestamps to millisecond integers (freqtrade feather convention)
    date_ms = (
        pd.to_datetime(timestamps).astype("int64") // 1_000_000
    )

    df = pd.DataFrame(
        {
            "date": date_ms.astype("int64"),
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": fair_values,
            "volume": np.ones(n, dtype=float),
        }
    )
    return df

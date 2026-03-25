"""Alpha factor: direct event-probability model.

Reads precomputed per-contract event probabilities (produced by
:func:`polymarket.event_model.predict_contract_probs`) and enriches the
contract OHLCV dataframe with the columns expected by
:class:`~user_data.strategies.DualModelPolymarketPortfolio`.

Columns added to the dataframe
-------------------------------
ml_fair_value    — P(BTC_T > K) from the event probability model.
ml_market_price  — Contract close price (= the ``close`` OHLCV column).
ml_edge          — ml_fair_value − ml_market_price.
ml_kelly_alloc   — Fractional Kelly allocation as a fraction of capital,
                   capped at ``max_alloc`` (further reduced for OTM contracts).
ml_h_remaining   — Floating hours until contract expiry.

Required metadata keys
-----------------------
``event_probs_df``  — DataFrame with a ``fair_value`` column, indexed by
                      UTC-aware timestamps (or with a ``dt_utc`` column that
                      will be set as the index).
``expiry_utc``      — Contract expiry as ISO-8601 UTC string.
``kelly_fraction``  — (optional, default 0.25) Fractional Kelly multiplier.
``min_edge``        — (optional, default 0.02) Minimum edge to allocate.
``max_alloc``       — (optional, default 0.30) Max per-contract allocation.

OTM distance penalty
---------------------
Applies only when BTC is **below** the strike (OTM for YES contracts).
``log_moneyness = log(BTC / K)``; the penalty is based on the **negative**
part only:

    moneyness_std = max(0, −log_moneyness) / σ_h

Contracts more than 1.5σ out-of-the-money are skipped entirely.
For contracts between 0 and 1.5σ OTM, min_edge is scaled up and max_alloc
scaled down proportionally:

    penalty = moneyness_std               # 0 at-the-money, 1.5 deep OTM
    effective_min_edge  = min_edge  × (1 + 2 × penalty)
    effective_max_alloc = max_alloc / (1 + penalty)

ITM contracts (BTC above strike, log_moneyness > 0) receive no penalty.

``log_moneyness`` and ``sigma_h`` are read from the ``event_probs_df`` index
(written by :func:`polymarket.event_model.predict_contract_probs`) and used
to compute the standardised moneyness at each bar.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas import DataFrame

from alpha.interface import IAlpha

logger = logging.getLogger(__name__)

# Standardised moneyness beyond which contracts are skipped as deep OTM.
OTM_SKIP_SIGMA: float = 1.5


class EventProbAlpha(IAlpha):
    """Direct event-probability alpha for Polymarket binary contracts.

    Usage inside a strategy::

        alpha = EventProbAlpha(dataframe, metadata)
        dataframe = alpha.process()
    """

    def process(self) -> DataFrame:
        df = self.dataframe.copy()
        meta = self.metadata

        expiry_utc: str = meta["expiry_utc"]
        event_probs: pd.DataFrame = meta["event_probs_df"]
        kelly_fraction: float = meta.get("kelly_fraction", 0.25)
        min_edge: float = meta.get("min_edge", 0.02)
        max_alloc: float = meta.get("max_alloc", 0.30)

        expiry_ts = pd.Timestamp(expiry_utc, tz="UTC")

        # ---- Hours remaining until expiry ----
        df["ml_h_remaining"] = df["date"].apply(
            lambda dt: max(0.0, (expiry_ts - dt).total_seconds() / 3600.0)
        )

        # ---- Align event probabilities to the contract dataframe ----
        # Normalise index: ensure event_probs is indexed by UTC-aware timestamps.
        probs = event_probs.copy()
        if not isinstance(probs.index, pd.DatetimeIndex):
            if "dt_utc" in probs.columns:
                probs = probs.set_index("dt_utc")
            else:
                raise KeyError(
                    "event_probs_df must be indexed by UTC timestamps or "
                    "have a 'dt_utc' column."
                )
        if probs.index.tzinfo is None:
            probs.index = probs.index.tz_localize("UTC")

        fair_aligned = probs["fair_value"].reindex(df["date"])
        # Forward-fill any small gaps from timestamp misalignment between the
        # BTC predictions CSV and the contract feather file.
        fair_aligned = fair_aligned.ffill()

        df["ml_fair_value"] = fair_aligned.values
        df["ml_market_price"] = df["close"]
        df["ml_edge"] = df["ml_fair_value"] - df["ml_market_price"]

        # ---- Fractional Kelly allocation with OTM penalty ----
        # Single-contract Kelly:  f = (p·b − q) / b
        # where b = (1/price − 1) = net odds, p = model prob, q = 1 − p.
        #
        # OTM penalty: standardised moneyness = log_moneyness / sigma_h
        # (available as columns if the event_probs_df was built with
        # add_contract_features; fall back to 0 if absent).
        log_moneyness_series = probs.get("log_moneyness")
        sigma_h_series = probs.get("sigma_h")

        kelly_allocs: list[float] = []
        for idx, row in df.iterrows():
            p = row["ml_fair_value"]
            price = row["ml_market_price"]

            if pd.isna(p) or pd.isna(price) or price <= 0.001 or price >= 0.999:
                kelly_allocs.append(0.0)
                continue

            # ---- OTM distance penalty ----
            # log_moneyness = log(BTC / K):
            #   > 0  →  BTC above strike  →  ITM  (no penalty)
            #   < 0  →  BTC below strike  →  OTM  (penalise)
            # We only scale up min_edge / scale down max_alloc when the contract
            # is out-of-the-money, i.e. when log_moneyness < 0.
            moneyness_std = 0.0
            if log_moneyness_series is not None and sigma_h_series is not None:
                dt = row["date"]
                if dt in log_moneyness_series.index and dt in sigma_h_series.index:
                    lm = log_moneyness_series.loc[dt]
                    sh = sigma_h_series.loc[dt]
                    if pd.notna(lm) and pd.notna(sh) and sh > 0:
                        # max(0, -lm): positive only when OTM (BTC below K)
                        moneyness_std = max(0.0, -lm) / sh

            # Skip deep OTM contracts entirely.
            if moneyness_std > OTM_SKIP_SIGMA:
                kelly_allocs.append(0.0)
                continue

            # Scale min_edge up and max_alloc down by OTM distance.
            penalty = max(0.0, moneyness_std)
            effective_min_edge = min_edge * (1.0 + 2.0 * penalty)
            effective_max_alloc = max_alloc / (1.0 + penalty)

            b = (1.0 / price) - 1.0
            q = 1.0 - p
            raw_kelly = (p * b - q) / b if b > 0 else 0.0

            if raw_kelly <= 0 or row["ml_edge"] < effective_min_edge:
                kelly_allocs.append(0.0)
            else:
                kelly_allocs.append(min(raw_kelly * kelly_fraction, effective_max_alloc))

        df["ml_kelly_alloc"] = kelly_allocs

        return df

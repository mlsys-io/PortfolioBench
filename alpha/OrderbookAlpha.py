"""Orderbook-derived alpha for Polymarket contracts.

Loads a pre-computed feature parquet (from run_pipeline.py), backward-fills
the latest orderbook snapshot onto each OHLCV candle, and adds two columns:

    ob_imbalance    : raw top-3-level orderbook imbalance [-1, 1]
    ob_imbalance_ema: EMA-smoothed version (span=ema_span candles)

Token-id is resolved automatically from the pair name using:
    freqtrade_pair_mapping.csv  →  condition_id
    markets.parquet             →  market_id
    tokens.parquet              →  token_id
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.interface import IAlpha

_DATA_DIR    = Path(__file__).resolve().parents[1] / "user_data" / "data" / "polymarket"
_FEATURE_DIR = _DATA_DIR / "feat_orderbook"

# Cached once per process: pair_base → token_id
_pair_token_map: dict[str, str] | None = None


def _build_pair_token_map() -> dict[str, str]:
    mapping = pd.read_csv(_DATA_DIR / "freqtrade_pair_mapping.csv")
    markets  = pd.read_parquet(_DATA_DIR / "markets.parquet", columns=["market_id", "condition_id"])
    tokens   = pd.read_parquet(_DATA_DIR / "tokens.parquet",  columns=["token_id", "market_id", "outcome"])

    cond_to_market = dict(zip(markets["condition_id"].str.lower(), markets["market_id"].astype(str)))
    tok_idx = {
        (str(r.market_id), r.outcome.strip().lower()): str(r.token_id)
        for r in tokens.itertuples(index=False)
    }

    result: dict[str, str] = {}
    for _, row in mapping.iterrows():
        cond_id   = str(row["Original_Condition_ID"]).lower()
        market_id = cond_to_market.get(cond_id)
        if not market_id:
            continue

        # "SomePairYES20250430_USDC-4h.feather" → pair_base = "SomePairYES20250430"
        stem = re.sub(r"-\d+[mhd]$", "", str(row["New_Filename"]).removesuffix(".feather"))
        if "_" not in stem:
            continue
        pair_base = stem.rsplit("_", 1)[0]

        outcome_key = "yes" if "YES" in pair_base.upper() else "no" if "NO" in pair_base.upper() else None
        if not outcome_key:
            continue

        token_id = tok_idx.get((market_id, outcome_key))
        if token_id:
            result[pair_base] = token_id

    return result


def _lookup_token_id(pair: str) -> str:
    global _pair_token_map
    if _pair_token_map is None:
        _pair_token_map = _build_pair_token_map()
    return _pair_token_map.get(pair.split("/")[0], "")


class OrderbookAlpha(IAlpha):
    def __init__(self, dataframe: pd.DataFrame, metadata: dict = None, ema_span: int = 8):
        self.ema_span = ema_span
        super().__init__(dataframe, metadata)

    def process(self) -> pd.DataFrame:
        df = self.dataframe
        token_id = self.metadata.get("token_id") or _lookup_token_id(self.metadata.get("pair", ""))

        feat_path = _FEATURE_DIR / f"feat_{token_id}.parquet"
        if not token_id or not feat_path.exists():
            df["ob_imbalance"] = np.nan
            df["ob_imbalance_ema"] = np.nan
            return df

        feat = pd.read_parquet(feat_path, columns=["snapshot_time", "imbalance_3"])
        feat["snapshot_time"] = pd.to_datetime(feat["snapshot_time"]).dt.tz_localize(None)
        feat = feat.sort_values("snapshot_time").reset_index(drop=True)

        candle_dates = pd.to_datetime(df["date"]).dt.tz_localize(None)
        order = np.argsort(candle_dates.values)
        left = pd.DataFrame({"date": candle_dates.iloc[order].values, "_idx": order})
        merged = (
            pd.merge_asof(left, feat.rename(columns={"snapshot_time": "date"}), on="date", direction="backward")
            .sort_values("_idx")
            .reset_index(drop=True)
        )

        imb = merged["imbalance_3"].fillna(0.0)
        df["ob_imbalance"] = imb.values
        df["ob_imbalance_ema"] = imb.ewm(span=self.ema_span, adjust=False).mean().values
        return df

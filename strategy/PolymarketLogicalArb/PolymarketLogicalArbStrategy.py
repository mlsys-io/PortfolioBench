from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

"""
This strategy implements a logical arbitrage approach over Polymarket Bitcoin
threshold contracts of the form "Will the price of Bitcoin be above X".

Core idea:
Contracts with the same expiry are linked by a deterministic logical ordering.
If Bitcoin is above a higher strike, then it must also be above every lower
strike. For example, if "BTC > 960" is true, then "BTC > 900" must also be true.
Because of this, the higher-strike YES contract is treated as the "subset" and
the lower-strike YES contract is treated as the "superset".

How relationships are built:
The strategy loads all BTC Above YES contracts from the pair source file,
groups them by expiry, sorts them by strike, and constructs pairwise logical
relationships. Each higher strike contract is mapped as implying each lower
strike contract with the same expiry.

How mispricing is measured:
For each contract, the strategy looks at all logical relationships involving that
pair. It aligns the close prices of the related contracts in time, computes the
gap

    gap = superset_price - subset_price

and then computes a rolling z-score of that gap. A low gap or very negative
gap z-score suggests the lower-strike contract may be underpriced relative to
the higher-strike contract, which is a potential logical inconsistency.

Signal construction:
Among all candidate relationships for a pair, the strategy selects the most
extreme mispricing using the lowest score, where score is the gap z-score when
available, otherwise the raw gap. It then ranks that score across the pair's
history to form a percentile-based signal.

Entry logic:
The strategy only enters long positions in contracts that are acting as the
logical "superset" in a relationship, meaning the lower-strike YES contract.
An entry is triggered when:
1. the pair is in the superset role,
2. enough history is available,
3. the rank percentile of the mispricing is low enough,
4. the contract is not too close to the end of the dataset,
5. recent momentum and jump filters do not indicate unstable price action,
6. the contract price is within the configured tradable range.

This means the strategy is trying to buy lower-strike contracts when they look
unusually cheap relative to logically implied higher-strike contracts.

Exit logic:
The strategy exits when the mispricing has largely normalized, represented by a
high rank percentile, or when the contract is near the end of the dataset.
This reflects the idea that once the logical inconsistency has mean-reverted,
the trade thesis is no longer attractive.

Risk and execution design:
- ROI is effectively disabled with a very large immediate ROI target.
- Stoploss is set very loose.
- Trailing stop is disabled.
This is done so that trades are mainly governed by the logical entry and exit
framework rather than being closed immediately by generic profit-taking rules.

In summary:
This is a structure-based prediction market strategy that combines deterministic
logical constraints with statistical ranking. It attempts to capture temporary
mispricings between related BTC threshold contracts by buying lower-strike YES
contracts when they appear too cheap relative to higher-strike YES contracts
that logically imply them.
"""


class PolymarketLogicalArbStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {"0": 100, "240": 0.0}
    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 8

    timeframe = "4h"

    # ------------------------------------------------------------------
    # Pair source settings
    # ------------------------------------------------------------------
    PAIR_SOURCE_PATH: str = "./strategy/PolymarketLogicalArb/freqtrade_pair_mapping.csv"
    PAIR_COLUMN: Optional[str] = None

    BTC_ABOVE_YES_PAIR_PATTERN = re.compile(
        r"^WillThePriceOfBitcoinBeAbove(\d+)YES(\d+(?:V\d+)?)\/USDC$",
        re.IGNORECASE,
    )

    BTC_ABOVE_FILE_PATTERN = re.compile(
        r"WillThePriceOfBitcoinBeAbove(\d+)(YES|NO)(\d+(?:V\d+)?)_USDC(?:-[A-Za-z0-9]+)?(?:\.feather)?$",
        re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Ranking-based settings
    # ------------------------------------------------------------------
    gap_window: int = 6
    min_periods_for_z: int = 4
    min_history_bars: int = 6
    min_bars_from_end: int = 1

    max_subset_mom_1bar: float = 0.50
    max_subset_mom_2bar: float = 0.80
    min_superset_mom_1bar: float = -0.25

    max_current_abs_ret_1bar: float = 0.80
    max_subset_abs_ret_1bar: float = 0.80
    max_superset_abs_ret_1bar: float = 0.80

    min_price: float = 0.01
    max_price: float = 0.99

    entry_rank_threshold: float = 0.30
    strong_entry_rank_threshold: float = 0.10

    # Kept for debugging visibility only
    max_gap_for_entry: float = 0.08
    max_gap_z_for_entry: float = 0.75

    # Much looser exits so ranked entries are not cancelled immediately
    exit_gap_threshold: float = 0.70
    exit_gap_z_threshold: float = 1.00
    exit_rank_threshold: float = 0.85

    DEBUG_PRINTS: bool = True

    RELATIONSHIPS: List[Dict[str, str]] = []
    _PAIR_CACHE_BUILT: bool = False

    # ------------------------------------------------------------------
    # Source loading helpers
    # ------------------------------------------------------------------
    @classmethod
    def _infer_pair_column(cls, df: pd.DataFrame) -> str:
        if cls.PAIR_COLUMN and cls.PAIR_COLUMN in df.columns:
            return cls.PAIR_COLUMN

        for col in df.columns:
            s = df[col].astype(str)
            if s.str.contains("BitcoinBeAbove", regex=False, na=False).any():
                return col

        return df.columns[0]

    @classmethod
    def _normalize_raw_entry_to_pair(cls, raw: str) -> Optional[str]:
        if raw is None:
            return None

        s = str(raw).strip()
        s = s.strip().strip(",").strip().strip('"').strip("'")
        if not s:
            return None

        if cls.BTC_ABOVE_YES_PAIR_PATTERN.match(s):
            return s

        m = cls.BTC_ABOVE_FILE_PATTERN.search(s)
        if m:
            strike = m.group(1)
            side = m.group(2).upper()
            expiry = m.group(3)
            return f"WillThePriceOfBitcoinBeAbove{strike}{side}{expiry}/USDC"

        if "_USDC" in s and "/USDC" not in s:
            s2 = re.sub(
                r"_USDC(?:-[A-Za-z0-9]+)?(?:\.feather)?$",
                "/USDC",
                s,
                flags=re.IGNORECASE,
            )
            m2 = re.search(
                r"WillThePriceOfBitcoinBeAbove(\d+)(YES|NO)(\d+(?:V\d+)?)\/USDC$",
                s2,
                flags=re.IGNORECASE,
            )
            if m2:
                strike = m2.group(1)
                side = m2.group(2).upper()
                expiry = m2.group(3)
                return f"WillThePriceOfBitcoinBeAbove{strike}{side}{expiry}/USDC"

        return None

    @classmethod
    def _load_raw_entries(cls, source_path: str) -> List[str]:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find pair source at {source_path!r}. "
                "Update PAIR_SOURCE_PATH in the strategy."
            )

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            col = cls._infer_pair_column(df)
            return df[col].astype(str).tolist()

        text = path.read_text(encoding="utf-8", errors="ignore")
        return [line.strip() for line in text.splitlines() if line.strip()]

    @classmethod
    def build_btc_relations_from_source(cls, source_path: str) -> List[Dict[str, str]]:
        raw_entries = cls._load_raw_entries(source_path)

        parsed: List[Tuple[str, int, str]] = []
        for raw in raw_entries:
            pair = cls._normalize_raw_entry_to_pair(raw)
            if not pair:
                continue

            m = cls.BTC_ABOVE_YES_PAIR_PATTERN.match(pair)
            if not m:
                continue

            strike = int(m.group(1))
            expiry = m.group(2)
            parsed.append((pair, strike, expiry))

        if not parsed:
            raise ValueError("No BTC Above YES /USDC contracts found in source file.")

        grouped: Dict[str, List[Tuple[str, int]]] = {}
        for pair, strike, expiry in parsed:
            grouped.setdefault(expiry, []).append((pair, strike))

        relations: List[Dict[str, str]] = []

        for expiry, items in grouped.items():
            uniq: Dict[str, int] = {}
            for pair, strike in items:
                uniq[pair] = strike

            items_sorted = sorted(uniq.items(), key=lambda x: x[1])

            for i in range(len(items_sorted)):
                lower_pair, lower_strike = items_sorted[i]
                for j in range(i + 1, len(items_sorted)):
                    higher_pair, higher_strike = items_sorted[j]
                    relations.append(
                        {
                            "id": f"btc_above_{higher_strike}_implies_above_{lower_strike}_{expiry}",
                            "subset_yes": higher_pair,
                            "superset_yes": lower_pair,
                        }
                    )

        return relations

    @classmethod
    def _ensure_relations_loaded(cls) -> None:
        if not cls._PAIR_CACHE_BUILT:
            cls.RELATIONSHIPS = cls.build_btc_relations_from_source(cls.PAIR_SOURCE_PATH)
            cls._PAIR_CACHE_BUILT = True

    @classmethod
    def _all_pairs(cls) -> List[str]:
        cls._ensure_relations_loaded()
        pairs: List[str] = []
        for rel in cls.RELATIONSHIPS:
            for pair in (rel["subset_yes"], rel["superset_yes"]):
                if pair not in pairs:
                    pairs.append(pair)
        return pairs

    @classmethod
    def _rels_for_pair(cls, pair: str) -> List[Dict[str, str]]:
        cls._ensure_relations_loaded()
        return [
            rel for rel in cls.RELATIONSHIPS
            if pair in (rel["subset_yes"], rel["superset_yes"])
        ]

    def informative_pairs(self):
        return [(pair, self.timeframe) for pair in self._all_pairs()]

    def _aligned_close_series(
        self,
        pair: str,
        base_dates: pd.Index,
        current_pair: str,
        current_df: pd.DataFrame,
    ) -> pd.Series:
        if pair == current_pair:
            return pd.Series(current_df["close"].to_numpy(), index=current_df.index)

        if self.dp is None:
            return pd.Series(np.nan, index=current_df.index)

        related = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
        if related is None or related.empty or "close" not in related.columns:
            return pd.Series(np.nan, index=current_df.index)

        related = related.copy()
        related = related[["date", "close"]].dropna(subset=["date"])
        related = related.sort_values("date").set_index("date")

        aligned = related["close"].reindex(base_dates, method="ffill")
        return pd.Series(aligned.to_numpy(), index=current_df.index)

    def _compute_relation_metrics(
        self,
        dataframe: pd.DataFrame,
        metadata: dict,
    ) -> pd.DataFrame:
        pair = metadata["pair"]
        rels = self._rels_for_pair(pair)

        dataframe["logic_signal"] = 0
        dataframe["logic_rel_id"] = None
        dataframe["logic_role"] = None
        dataframe["logic_best_gap"] = np.nan
        dataframe["logic_best_gap_z"] = np.nan
        dataframe["logic_subset_mom_1"] = np.nan
        dataframe["logic_subset_mom_2"] = np.nan
        dataframe["logic_superset_mom_1"] = np.nan
        dataframe["logic_current_abs_ret_1"] = np.nan
        dataframe["logic_subset_abs_ret_1"] = np.nan
        dataframe["logic_superset_abs_ret_1"] = np.nan
        dataframe["logic_rank_score"] = np.nan
        dataframe["logic_rank_pct"] = np.nan
        dataframe["bars_to_end"] = np.nan
        dataframe["enough_history"] = 0

        dataframe["dbg_rank_entry"] = 0
        dataframe["dbg_score_sanity"] = 0
        dataframe["dbg_subset_not_ripping"] = 0
        dataframe["dbg_superset_not_collapsing"] = 0
        dataframe["dbg_jump_veto_pass"] = 0
        dataframe["dbg_not_near_end"] = 0
        dataframe["dbg_tradeable_preview"] = 0
        dataframe["dbg_reject_reason"] = ""

        if not rels:
            dataframe["dbg_reject_reason"] = "no_relation"
            return dataframe

        base = dataframe.copy()
        base_dates = pd.Index(base["date"])

        best_score = pd.Series(np.inf, index=dataframe.index)
        best_gap = pd.Series(np.nan, index=dataframe.index)
        best_gap_z = pd.Series(np.nan, index=dataframe.index)
        best_subset_m1 = pd.Series(np.nan, index=dataframe.index)
        best_subset_m2 = pd.Series(np.nan, index=dataframe.index)
        best_superset_m1 = pd.Series(np.nan, index=dataframe.index)
        best_curr_abs1 = pd.Series(np.nan, index=dataframe.index)
        best_subset_abs1 = pd.Series(np.nan, index=dataframe.index)
        best_superset_abs1 = pd.Series(np.nan, index=dataframe.index)
        best_rel_id = pd.Series([None] * len(dataframe), index=dataframe.index, dtype="object")
        best_role = pd.Series([None] * len(dataframe), index=dataframe.index, dtype="object")

        current_abs1 = (
            dataframe["close"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .abs()
            .fillna(0.0)
        )

        for rel in rels:
            subset_s = self._aligned_close_series(
                pair=rel["subset_yes"],
                base_dates=base_dates,
                current_pair=pair,
                current_df=base,
            )
            superset_s = self._aligned_close_series(
                pair=rel["superset_yes"],
                base_dates=base_dates,
                current_pair=pair,
                current_df=base,
            )

            gap = pd.Series(
                superset_s.to_numpy() - subset_s.to_numpy(),
                index=dataframe.index,
            )

            gap_mean = gap.rolling(self.gap_window, min_periods=self.min_periods_for_z).mean()
            gap_std = gap.rolling(self.gap_window, min_periods=self.min_periods_for_z).std()
            gap_z = (gap - gap_mean) / gap_std.replace(0.0, np.nan)

            subset_m1 = subset_s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            subset_m2 = subset_s.pct_change(2).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            superset_m1 = superset_s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

            subset_abs1 = subset_m1.abs()
            superset_abs1 = superset_m1.abs()

            score = gap_z.fillna(gap)

            mask = score < best_score

            best_score = best_score.where(~mask, score)
            best_gap = best_gap.where(~mask, gap)
            best_gap_z = best_gap_z.where(~mask, gap_z)
            best_subset_m1 = best_subset_m1.where(~mask, subset_m1)
            best_subset_m2 = best_subset_m2.where(~mask, subset_m2)
            best_superset_m1 = best_superset_m1.where(~mask, superset_m1)
            best_curr_abs1 = best_curr_abs1.where(~mask, current_abs1)
            best_subset_abs1 = best_subset_abs1.where(~mask, subset_abs1)
            best_superset_abs1 = best_superset_abs1.where(~mask, superset_abs1)
            best_rel_id = best_rel_id.where(~mask, rel["id"])

            role = "subset" if pair == rel["subset_yes"] else "superset"
            role_series = pd.Series([role] * len(dataframe), index=dataframe.index)
            best_role = best_role.where(~mask, role_series)

        dataframe["logic_rel_id"] = best_rel_id
        dataframe["logic_role"] = best_role
        dataframe["logic_best_gap"] = best_gap
        dataframe["logic_best_gap_z"] = best_gap_z
        dataframe["logic_subset_mom_1"] = best_subset_m1
        dataframe["logic_subset_mom_2"] = best_subset_m2
        dataframe["logic_superset_mom_1"] = best_superset_m1
        dataframe["logic_current_abs_ret_1"] = best_curr_abs1
        dataframe["logic_subset_abs_ret_1"] = best_subset_abs1
        dataframe["logic_superset_abs_ret_1"] = best_superset_abs1
        dataframe["logic_rank_score"] = best_score

        valid_rank = dataframe["logic_rank_score"].notna()
        dataframe.loc[valid_rank, "logic_rank_pct"] = (
            dataframe.loc[valid_rank, "logic_rank_score"].rank(pct=True, method="average")
        )

        bars_remaining = (len(dataframe) - 1) - np.arange(len(dataframe))
        dataframe["bars_to_end"] = bars_remaining
        dataframe["enough_history"] = (np.arange(len(dataframe)) >= self.min_history_bars).astype(int)

        subset_not_ripping = (
            (dataframe["logic_subset_mom_1"] <= self.max_subset_mom_1bar)
            & (dataframe["logic_subset_mom_2"] <= self.max_subset_mom_2bar)
        )

        superset_not_collapsing = (
            dataframe["logic_superset_mom_1"] >= self.min_superset_mom_1bar
        )

        jump_veto_pass = (
            (dataframe["logic_current_abs_ret_1"] <= self.max_current_abs_ret_1bar)
            & (dataframe["logic_subset_abs_ret_1"] <= self.max_subset_abs_ret_1bar)
            & (dataframe["logic_superset_abs_ret_1"] <= self.max_superset_abs_ret_1bar)
        )

        not_near_end = dataframe["bars_to_end"] > self.min_bars_from_end

        rank_entry = (
            dataframe["logic_rank_pct"] <= self.entry_rank_threshold
        )

        score_sanity = (
            (dataframe["logic_best_gap"] <= self.max_gap_for_entry)
            & (
                dataframe["logic_best_gap_z"].isna()
                | (dataframe["logic_best_gap_z"] <= self.max_gap_z_for_entry)
            )
        )

        dataframe["logic_signal"] = (
            (dataframe["logic_role"] == "superset")
            & (dataframe["enough_history"] == 1)
            & rank_entry
            & subset_not_ripping
            & superset_not_collapsing
            & jump_veto_pass
            & not_near_end
        ).astype(int)

        dataframe["dbg_rank_entry"] = rank_entry.astype(int)
        dataframe["dbg_score_sanity"] = score_sanity.astype(int)
        dataframe["dbg_subset_not_ripping"] = subset_not_ripping.astype(int)
        dataframe["dbg_superset_not_collapsing"] = superset_not_collapsing.astype(int)
        dataframe["dbg_jump_veto_pass"] = jump_veto_pass.astype(int)
        dataframe["dbg_not_near_end"] = not_near_end.astype(int)
        dataframe["dbg_tradeable_preview"] = (
            (dataframe["close"] > self.min_price)
            & (dataframe["close"] < self.max_price)
        ).astype(int)

        reasons = []
        for _, row in dataframe.iterrows():
            if row["logic_signal"] == 1:
                reasons.append("entered")
                continue

            r = []
            if row["logic_role"] != "superset":
                r.append("not_superset")
            if row["enough_history"] != 1:
                r.append("no_history")
            if row["dbg_rank_entry"] != 1:
                r.append("rank_fail")
            if row["dbg_score_sanity"] != 1:
                r.append("score_fail")
            if row["dbg_subset_not_ripping"] != 1:
                r.append("subset_mom_fail")
            if row["dbg_superset_not_collapsing"] != 1:
                r.append("superset_mom_fail")
            if row["dbg_jump_veto_pass"] != 1:
                r.append("jump_fail")
            if row["dbg_not_near_end"] != 1:
                r.append("near_end")
            if row["dbg_tradeable_preview"] != 1:
                r.append("price_fail")

            reasons.append("|".join(r) if r else "unknown")

        dataframe["dbg_reject_reason"] = reasons

        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self._compute_relation_metrics(dataframe, metadata)

        dataframe["contract_tradeable"] = (
            (dataframe["close"] > self.min_price)
            & (dataframe["close"] < self.max_price)
        ).astype(int)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["logic_signal"] = (
            (dataframe["logic_role"] == "superset")
            & (dataframe["enough_history"] == 1)
            & (dataframe["logic_rank_pct"] <= self.entry_rank_threshold)
            & (dataframe["bars_to_end"] > self.min_bars_from_end)
        ).astype(int)

        entry_mask = (
            (dataframe["logic_signal"] == 1)
            & (dataframe["contract_tradeable"] == 1)
        )

        dataframe.loc[entry_mask, "enter_long"] = 1

        strong_mask = entry_mask & (
            dataframe["logic_rank_pct"] <= self.strong_entry_rank_threshold
        )

        dataframe.loc[entry_mask, "enter_tag"] = (
            "logic_v2:"
            + dataframe["logic_rel_id"].astype(str)
            + ":gap="
            + dataframe["logic_best_gap"].round(4).astype(str)
            + ":z="
            + dataframe["logic_best_gap_z"].round(2).astype(str)
            + ":rank="
            + dataframe["logic_rank_pct"].round(3).astype(str)
        )

        dataframe.loc[strong_mask, "enter_tag"] = (
            "logic_v2_strong:"
            + dataframe["logic_rel_id"].astype(str)
            + ":gap="
            + dataframe["logic_best_gap"].round(4).astype(str)
            + ":z="
            + dataframe["logic_best_gap_z"].round(2).astype(str)
            + ":rank="
            + dataframe["logic_rank_pct"].round(3).astype(str)
        )

        if self.DEBUG_PRINTS:
            pair = metadata["pair"]
            try:
                total_rows = len(dataframe)
                superset_rows = int((dataframe["logic_role"] == "superset").sum())
                history_rows = int((dataframe["enough_history"] == 1).sum())
                rank_rows = int((dataframe["dbg_rank_entry"] == 1).sum())
                score_rows = int((dataframe["dbg_score_sanity"] == 1).sum())
                subset_rows = int((dataframe["dbg_subset_not_ripping"] == 1).sum())
                superset_ok_rows = int((dataframe["dbg_superset_not_collapsing"] == 1).sum())
                jump_rows = int((dataframe["dbg_jump_veto_pass"] == 1).sum())
                near_end_rows = int((dataframe["dbg_not_near_end"] == 1).sum())
                price_rows = int((dataframe["contract_tradeable"] == 1).sum())
                entry_rows = int(entry_mask.sum())

                same_candle_exit = (
                    (dataframe["logic_best_gap"] >= self.exit_gap_threshold)
                    | (dataframe["logic_best_gap_z"] >= self.exit_gap_z_threshold)
                    | (dataframe["logic_rank_pct"] >= self.exit_rank_threshold)
                    | (dataframe["close"] >= 0.98)
                    | (dataframe["close"] <= 0.01)
                    | (dataframe["bars_to_end"] <= 1)
                    | (dataframe["logic_current_abs_ret_1"] > self.max_current_abs_ret_1bar)
                )

                print("\n" + "=" * 90)
                print(f"[DEBUG] Pair: {pair}")
                print(f"[DEBUG] rows={total_rows}")
                print(f"[DEBUG] superset_rows={superset_rows}")
                print(f"[DEBUG] enough_history_rows={history_rows}")
                print(f"[DEBUG] rank_pass_rows={rank_rows}")
                print(f"[DEBUG] score_pass_rows={score_rows}")
                print(f"[DEBUG] subset_momentum_pass_rows={subset_rows}")
                print(f"[DEBUG] superset_momentum_pass_rows={superset_ok_rows}")
                print(f"[DEBUG] jump_veto_pass_rows={jump_rows}")
                print(f"[DEBUG] not_near_end_rows={near_end_rows}")
                print(f"[DEBUG] tradeable_rows={price_rows}")
                print(f"[DEBUG] final_entry_rows={entry_rows}")
                print(f"[DEBUG] entry_and_exit_same_candle_rows={int((entry_mask & same_candle_exit).sum())}")

                if entry_rows == 0:
                    print("[DEBUG] Top 10 rejection reasons:")
                    print(dataframe["dbg_reject_reason"].value_counts(dropna=False).head(10).to_string())
                else:
                    cols = [
                        "date",
                        "close",
                        "logic_rel_id",
                        "logic_best_gap",
                        "logic_best_gap_z",
                        "logic_rank_pct",
                        "enter_tag",
                    ]
                    print("[DEBUG] Entry rows:")
                    print(dataframe.loc[entry_mask, cols].tail(20).to_string(index=False))

                print("=" * 90 + "\n")
            except Exception as e:
                print(f"[DEBUG] Failed to print debug summary for {pair}: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        # Delay exits by 1 candle
        dataframe["can_exit"] = dataframe["bars_to_end"] < (len(dataframe) - 2)

        exit_mask = (
            (dataframe["logic_rank_pct"] >= 0.85)
            | (dataframe["bars_to_end"] <= 1)
        )

        dataframe.loc[exit_mask, "exit_long"] = 1
        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        if pair not in self._all_pairs():
            return False

        if rate < self.min_price or rate > self.max_price:
            return False

        return True
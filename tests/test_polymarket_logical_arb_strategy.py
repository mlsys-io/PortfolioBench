"""Tests for PolymarketLogicalArbStrategy."""

# to test this file, run the command on bash: python -m pytest tests/test_polymarket_logical_arb_strategy.py -q

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from strategy.PolymarketLogicalArb.PolymarketLogicalArbStrategy import PolymarketLogicalArbStrategy

def _make_pair_mapping_csv(tmp_path: Path) -> Path:
    """Create a minimal pair mapping CSV for BTC threshold contracts."""
    pairs = [
        "WillThePriceOfBitcoinBeAbove900YES20260121/USDC",
        "WillThePriceOfBitcoinBeAbove920YES20260121/USDC",
        "WillThePriceOfBitcoinBeAbove960YES20260121/USDC",
    ]
    path = tmp_path / "freqtrade_pair_mapping.csv"
    pd.DataFrame({"pair": pairs}).to_csv(path, index=False)
    return path


def _make_strategy_ohlcv(n: int = 40, start: str = "2026-01-21") -> pd.DataFrame:
    """Create a minimal OHLCV dataframe with deterministic 4h candles."""
    dates = pd.date_range(start, periods=n, freq="4H", tz="UTC")
    close = np.linspace(0.30, 0.45, n)

    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": np.full(n, 1000.0),
        }
    )


class DummyDP:
    """Minimal dataprovider stub for related pair lookups."""

    def __init__(self, pair_to_df: dict[str, pd.DataFrame]):
        self.pair_to_df = pair_to_df

    def get_pair_dataframe(self, pair: str, timeframe: str):
        return self.pair_to_df.get(pair)


class TestPolymarketLogicalArbStrategyHelpers:
    def test_normalize_raw_entry_to_pair_accepts_pair_format(self):
        raw = "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"
        pair = PolymarketLogicalArbStrategy._normalize_raw_entry_to_pair(raw)
        assert pair == raw

    def test_normalize_raw_entry_to_pair_converts_file_format(self):
        raw = "WillThePriceOfBitcoinBeAbove900YES20260121_USDC.feather"
        pair = PolymarketLogicalArbStrategy._normalize_raw_entry_to_pair(raw)
        assert pair == "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"

    def test_build_btc_relations_from_source(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)
        rels = PolymarketLogicalArbStrategy.build_btc_relations_from_source(str(csv_path))

        rel_ids = {r["id"] for r in rels}
        assert "btc_above_920_implies_above_900_20260121" in rel_ids
        assert "btc_above_960_implies_above_900_20260121" in rel_ids
        assert "btc_above_960_implies_above_920_20260121" in rel_ids
        assert len(rels) == 3

    def test_build_btc_relations_empty_source_raises(self, tmp_path):
        path = tmp_path / "pairs.csv"
        pd.DataFrame({"pair": ["ETH/USDC", "BTC/USDC"]}).to_csv(path, index=False)

        with pytest.raises(ValueError):
            PolymarketLogicalArbStrategy.build_btc_relations_from_source(str(path))


class TestPolymarketLogicalArbStrategyIndicators:
    def setup_method(self):
        PolymarketLogicalArbStrategy._PAIR_CACHE_BUILT = False
        PolymarketLogicalArbStrategy.RELATIONSHIPS = []

    def test_informative_pairs_returns_all_unique_pairs(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        pairs = strategy.informative_pairs()
        pair_names = {p[0] for p in pairs}

        assert "WillThePriceOfBitcoinBeAbove900YES20260121/USDC" in pair_names
        assert "WillThePriceOfBitcoinBeAbove920YES20260121/USDC" in pair_names
        assert "WillThePriceOfBitcoinBeAbove960YES20260121/USDC" in pair_names
        assert all(tf == strategy.timeframe for _, tf in pairs)

    def test_populate_indicators_adds_expected_columns(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        base = _make_strategy_ohlcv()
        related_920 = base.copy()
        related_960 = base.copy()

        related_920["close"] = np.linspace(0.20, 0.30, len(related_920))
        related_960["close"] = np.linspace(0.10, 0.20, len(related_960))

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy.dp = DummyDP(
            {
                "WillThePriceOfBitcoinBeAbove920YES20260121/USDC": related_920,
                "WillThePriceOfBitcoinBeAbove960YES20260121/USDC": related_960,
            }
        )
        strategy.DEBUG_PRINTS = False
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        result = strategy.populate_indicators(
            base.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        expected_cols = [
            "logic_signal",
            "logic_rel_id",
            "logic_role",
            "logic_best_gap",
            "logic_best_gap_z",
            "logic_subset_mom_1",
            "logic_subset_mom_2",
            "logic_superset_mom_1",
            "logic_current_abs_ret_1",
            "logic_subset_abs_ret_1",
            "logic_superset_abs_ret_1",
            "logic_rank_score",
            "logic_rank_pct",
            "bars_to_end",
            "enough_history",
            "contract_tradeable",
            "dbg_reject_reason",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_contract_tradeable_is_binary(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        base = _make_strategy_ohlcv()
        base.loc[0, "close"] = 0.005
        base.loc[1, "close"] = 0.995

        related_920 = _make_strategy_ohlcv()
        related_960 = _make_strategy_ohlcv()

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy.dp = DummyDP(
            {
                "WillThePriceOfBitcoinBeAbove920YES20260121/USDC": related_920,
                "WillThePriceOfBitcoinBeAbove960YES20260121/USDC": related_960,
            }
        )
        strategy.DEBUG_PRINTS = False
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        result = strategy.populate_indicators(
            base.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        assert set(result["contract_tradeable"].unique()).issubset({0, 1})
        assert result.loc[0, "contract_tradeable"] == 0
        assert result.loc[1, "contract_tradeable"] == 0


class TestPolymarketLogicalArbStrategySignals:
    def setup_method(self):
        PolymarketLogicalArbStrategy._PAIR_CACHE_BUILT = False
        PolymarketLogicalArbStrategy.RELATIONSHIPS = []

    def test_populate_entry_trend_adds_enter_columns(self):
        strategy = PolymarketLogicalArbStrategy({})
        strategy.DEBUG_PRINTS = False

        n = 12
        df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-21", periods=n, freq="4H", tz="UTC"),
                "close": np.full(n, 0.40),
                "logic_role": ["superset"] * n,
                "enough_history": [0] * 6 + [1] * 6,
                "logic_rank_pct": [0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.25, 0.09, 0.50, 0.20, 0.31, 0.05],
                "bars_to_end": list(reversed(range(n))),
                "contract_tradeable": [1] * n,
                "logic_rel_id": ["btc_above_960_implies_above_900_20260121"] * n,
                "logic_best_gap": np.linspace(0.05, 0.01, n),
                "logic_best_gap_z": np.linspace(-0.5, -2.0, n),
            }
        )

        result = strategy.populate_entry_trend(
            df.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        assert "enter_long" in result.columns
        assert "enter_tag" in result.columns
        assert (result["enter_long"].fillna(0) >= 0).all()
        assert result["enter_long"].fillna(0).sum() > 0

    def test_strong_entry_tag_is_used_for_very_low_rank(self):
        strategy = PolymarketLogicalArbStrategy({})
        strategy.DEBUG_PRINTS = False

        df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-21", periods=10, freq="4H", tz="UTC"),
                "close": np.full(10, 0.40),
                "logic_role": ["superset"] * 10,
                "enough_history": [1] * 10,
                "logic_rank_pct": [0.50, 0.40, 0.25, 0.09, 0.08, 0.30, 0.11, 0.10, 0.50, 0.70],
                "bars_to_end": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                "contract_tradeable": [1] * 10,
                "logic_rel_id": ["btc_above_960_implies_above_900_20260121"] * 10,
                "logic_best_gap": np.full(10, 0.03),
                "logic_best_gap_z": np.full(10, -1.5),
            }
        )

        result = strategy.populate_entry_trend(
            df.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        strong_tags = result["enter_tag"].dropna().astype(str)
        assert any(tag.startswith("logic_v2_strong:") for tag in strong_tags)

    def test_populate_exit_trend_adds_exit_long(self):
        strategy = PolymarketLogicalArbStrategy({})

        df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-21", periods=8, freq="4H", tz="UTC"),
                "logic_rank_pct": [0.10, 0.20, 0.30, 0.50, 0.86, 0.40, 0.20, 0.10],
                "bars_to_end": [7, 6, 5, 4, 3, 2, 1, 0],
            }
        )

        result = strategy.populate_exit_trend(
            df.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        assert "exit_long" in result.columns
        assert result["exit_long"].fillna(0).sum() >= 2

    def test_exit_triggered_by_high_rank_or_near_end(self):
        strategy = PolymarketLogicalArbStrategy({})

        df = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-21", periods=6, freq="4H", tz="UTC"),
                "logic_rank_pct": [0.10, 0.20, 0.86, 0.30, 0.20, 0.10],
                "bars_to_end": [5, 4, 3, 2, 1, 0],
            }
        )

        result = strategy.populate_exit_trend(
            df.copy(),
            metadata={"pair": "WillThePriceOfBitcoinBeAbove900YES20260121/USDC"},
        )

        assert result.loc[2, "exit_long"] == 1
        assert result.loc[4, "exit_long"] == 1
        assert result.loc[5, "exit_long"] == 1


class TestPolymarketLogicalArbStrategyEntryConfirmation:
    def setup_method(self):
        PolymarketLogicalArbStrategy._PAIR_CACHE_BUILT = False
        PolymarketLogicalArbStrategy.RELATIONSHIPS = []

    def test_confirm_trade_entry_accepts_valid_pair_and_price(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        ok = strategy.confirm_trade_entry(
            pair="WillThePriceOfBitcoinBeAbove900YES20260121/USDC",
            order_type="limit",
            amount=1000.0,
            rate=0.45,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag="logic_v2:test",
            side="long",
        )

        assert ok is True

    def test_confirm_trade_entry_rejects_unknown_pair(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        ok = strategy.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1000.0,
            rate=0.45,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag="logic_v2:test",
            side="long",
        )

        assert ok is False

    def test_confirm_trade_entry_rejects_bad_price(self, tmp_path):
        csv_path = _make_pair_mapping_csv(tmp_path)

        strategy = PolymarketLogicalArbStrategy({})
        strategy.PAIR_SOURCE_PATH = str(csv_path)
        strategy._PAIR_CACHE_BUILT = False
        strategy.RELATIONSHIPS = []

        low = strategy.confirm_trade_entry(
            pair="WillThePriceOfBitcoinBeAbove900YES20260121/USDC",
            order_type="limit",
            amount=1000.0,
            rate=0.001,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag="logic_v2:test",
            side="long",
        )

        high = strategy.confirm_trade_entry(
            pair="WillThePriceOfBitcoinBeAbove900YES20260121/USDC",
            order_type="limit",
            amount=1000.0,
            rate=1.10,
            time_in_force="GTC",
            current_time=datetime.now(timezone.utc),
            entry_tag="logic_v2:test",
            side="long",
        )

        assert low is False
        assert high is False
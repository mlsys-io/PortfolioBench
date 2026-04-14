"""Tests for polymarket.contracts — metadata loading and normalisation."""

import tempfile
from pathlib import Path

import pytest

from polymarket.contracts import (
    ContractMetadata,
    _make_pair,
    _parse_strike_direction,
    _settlement_from_outcome_prices,
    load_contracts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

JAN20_JSONL = Path(__file__).parents[2] / "user_data/data/polymarket_contracts/jan20.jsonl"


@pytest.fixture
def jan20_contracts():
    return load_contracts(JAN20_JSONL)


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers
# ---------------------------------------------------------------------------

class TestParseStrikeDirection:
    def test_above_with_comma(self):
        q = "Will the price of Bitcoin be above $90,000 on January 20?"
        strike, direction = _parse_strike_direction(q)
        assert strike == 90_000.0
        assert direction == "above"

    def test_above_no_comma(self):
        q = "Will the price of Bitcoin be above $84000 on January 20?"
        strike, direction = _parse_strike_direction(q)
        assert strike == 84_000.0
        assert direction == "above"

    def test_less_than(self):
        q = "Will the price of Bitcoin be less than $84,000 on January 20?"
        strike, direction = _parse_strike_direction(q)
        assert strike == 84_000.0
        assert direction == "below"

    def test_unrecognised_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_strike_direction("Something unrelated")


class TestSettlementFromOutcomePrices:
    def test_yes_wins(self):
        assert _settlement_from_outcome_prices('["1", "0"]') == 1.0

    def test_no_wins(self):
        assert _settlement_from_outcome_prices('["0", "1"]') == 0.0


class TestMakePair:
    def test_above_jan20_yes(self):
        pair = _make_pair(90_000, "above", "2026-01-20T17:00:00Z", "YES")
        assert pair == "BTCABOVE90K-JAN20-YES/USDT"

    def test_above_jan20_no(self):
        pair = _make_pair(90_000, "above", "2026-01-20T17:00:00Z", "NO")
        assert pair == "BTCABOVE90K-JAN20-NO/USDT"

    def test_below_jan20_yes(self):
        pair = _make_pair(84_000, "below", "2026-01-20T17:00:00Z", "YES")
        assert pair == "BTCBELOW84K-JAN20-YES/USDT"

    def test_strike_rounding(self):
        # 88000 → 88K
        pair = _make_pair(88_000, "above", "2026-01-20T17:00:00Z", "YES")
        assert "88K" in pair

    def test_three_digit_k(self):
        pair = _make_pair(100_000, "above", "2026-01-20T17:00:00Z", "YES")
        assert "100K" in pair


# ---------------------------------------------------------------------------
# Integration tests: load_contracts on real JSONL
# ---------------------------------------------------------------------------

class TestLoadContractsJan20:
    def test_loads_twelve_contracts(self, jan20_contracts):
        assert len(jan20_contracts) == 12

    def test_sorted_by_strike(self, jan20_contracts):
        strikes = [c.strike for c in jan20_contracts]
        assert strikes == sorted(strikes)

    def test_all_have_same_expiry(self, jan20_contracts):
        expiries = {c.end_date_utc for c in jan20_contracts}
        assert len(expiries) == 1
        assert "2026-01-20T17:00:00Z" in expiries

    def test_known_settlements(self, jan20_contracts):
        # Ground truth from outcomePrices + verified against BTC data
        expected = {
            84_000: 1.0,   # YES: 90064 > 84000
            86_000: 1.0,
            88_000: 1.0,
            90_000: 1.0,   # YES: 90064 > 90000
            92_000: 0.0,   # NO: 90064 < 92000
            94_000: 0.0,
            96_000: 0.0,
            98_000: 0.0,
            100_000: 0.0,
            102_000: 0.0,
            104_000: 0.0,
        }
        for c in jan20_contracts:
            if c.direction == "above" and c.strike in expected:
                assert c.settlement == expected[c.strike], (
                    f"Strike {c.strike}: expected settlement {expected[c.strike]}, "
                    f"got {c.settlement}"
                )

    def test_pair_format(self, jan20_contracts):
        for c in jan20_contracts:
            assert c.pair_yes.endswith("-YES/USDT"), c.pair_yes
            assert c.pair_no.endswith("-NO/USDT"), c.pair_no
            assert "USDT" in c.pair_yes

    def test_all_are_dataclass_instances(self, jan20_contracts):
        for c in jan20_contracts:
            assert isinstance(c, ContractMetadata)

    def test_volume_positive(self, jan20_contracts):
        for c in jan20_contracts:
            assert c.volume_usd > 0

    def test_below_contract_present(self, jan20_contracts):
        below = [c for c in jan20_contracts if c.direction == "below"]
        assert len(below) == 1
        assert below[0].strike == 84_000
        # "less than $84k" settled NO (BTC was above $84k)
        assert below[0].settlement == 0.0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_contracts("/nonexistent/path.jsonl")

    def test_malformed_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            fname = f.name
        with pytest.raises(ValueError, match="JSON parse error"):
            load_contracts(fname)

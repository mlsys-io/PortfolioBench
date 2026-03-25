"""Tests for polymarket.settlement — resolution price and settlement logic."""

from pathlib import Path

import pandas as pd
import pytest

from polymarket.contracts import load_contracts
from polymarket.settlement import (
    compute_settlement,
    get_resolution_price,
    load_btc_hourly,
    verify_settlements,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BTC_CSV = Path(__file__).parents[2] / "mycode/data/data_1h.csv"
JAN20_JSONL = Path(__file__).parents[2] / "user_data/data/polymarket_contracts/jan20.jsonl"

pytestmark = pytest.mark.skipif(
    not BTC_CSV.exists(),
    reason="data_1h.csv not found (run from repo root)",
)


@pytest.fixture(scope="module")
def btc_df():
    return load_btc_hourly(str(BTC_CSV))


@pytest.fixture(scope="module")
def jan20_contracts():
    return load_contracts(JAN20_JSONL)


# ---------------------------------------------------------------------------
# Unit tests: compute_settlement
# ---------------------------------------------------------------------------

class TestComputeSettlement:
    def test_above_yes(self):
        assert compute_settlement(90_064.0, 90_000.0, "above") == 1.0

    def test_above_no(self):
        assert compute_settlement(89_999.0, 90_000.0, "above") == 0.0

    def test_below_yes(self):
        assert compute_settlement(83_000.0, 84_000.0, "below") == 1.0

    def test_below_no(self):
        assert compute_settlement(85_000.0, 84_000.0, "below") == 0.0

    def test_exact_boundary_above(self):
        # Strictly greater-than: price == strike → NO
        assert compute_settlement(90_000.0, 90_000.0, "above") == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="Unknown direction"):
            compute_settlement(90_000.0, 90_000.0, "sideways")


# ---------------------------------------------------------------------------
# Integration tests: get_resolution_price
# ---------------------------------------------------------------------------

class TestGetResolutionPrice:
    def test_jan20_resolution_price(self, btc_df):
        # endDate for Jan-20 contracts is 2026-01-20T17:00:00Z
        price = get_resolution_price(btc_df, "2026-01-20T17:00:00Z")
        # The Open of the 17:00 UTC candle is 90,064
        assert price == pytest.approx(90_064.0, abs=1.0)

    def test_missing_candle_raises(self, btc_df):
        with pytest.raises(ValueError, match="No hourly candle found"):
            get_resolution_price(btc_df, "2099-01-01T00:00:00Z")

    def test_resolution_explains_outcomes(self, btc_df):
        price = get_resolution_price(btc_df, "2026-01-20T17:00:00Z")
        # Contracts resolved YES
        for strike in [84_000, 86_000, 88_000, 90_000]:
            assert price > strike, f"Expected price {price} > {strike}"
        # Contracts resolved NO
        for strike in [92_000, 94_000, 96_000, 98_000, 100_000, 102_000, 104_000]:
            assert price < strike, f"Expected price {price} < {strike}"


# ---------------------------------------------------------------------------
# Integration tests: verify_settlements
# ---------------------------------------------------------------------------

class TestVerifySettlements:
    def test_all_contracts_match(self, jan20_contracts, btc_df):
        results = verify_settlements(jan20_contracts, btc_df)
        assert len(results) == 12
        failures = [r for r in results if not r["match"]]
        assert failures == [], (
            f"Settlement mismatches:\n" +
            "\n".join(f"  {r['slug']}: outcome={r['outcome_prices_settlement']}, "
                      f"btc={r['btc_derived_settlement']}" for r in failures)
        )

    def test_result_structure(self, jan20_contracts, btc_df):
        results = verify_settlements(jan20_contracts, btc_df)
        required_keys = {
            "slug", "strike", "direction", "resolution_price",
            "outcome_prices_settlement", "btc_derived_settlement", "match",
        }
        for r in results:
            assert required_keys.issubset(r.keys())

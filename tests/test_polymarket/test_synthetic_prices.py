"""Tests for polymarket.synthetic_prices — OHLCV generation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket.contracts import load_contracts
from polymarket.settlement import load_btc_hourly
from polymarket.synthetic_prices import (
    PRICE_CEIL,
    PRICE_FLOOR,
    _calibrate_sigma,
    build_synthetic_ohlcv,
)

# ---------------------------------------------------------------------------
# Paths / marks
# ---------------------------------------------------------------------------

BTC_CSV = Path(__file__).parents[2] / "mycode/data/data_1h.csv"
JAN20_JSONL = Path(__file__).parents[2] / "user_data/data/polymarket_contracts/jan20.jsonl"

skip_if_no_data = pytest.mark.skipif(
    not BTC_CSV.exists(),
    reason="data_1h.csv not found",
)


@pytest.fixture(scope="module")
def btc_df():
    if not BTC_CSV.exists():
        pytest.skip("data_1h.csv not found")
    return load_btc_hourly(str(BTC_CSV))


@pytest.fixture(scope="module")
def jan20_contracts():
    return load_contracts(JAN20_JSONL)


# ---------------------------------------------------------------------------
# Unit tests: _calibrate_sigma
# ---------------------------------------------------------------------------

class TestCalibrateSigma:
    @skip_if_no_data
    def test_returns_positive_float(self, btc_df):
        ts = pd.Timestamp("2026-01-13 17:00:00", tz="UTC")
        sigma = _calibrate_sigma(btc_df, before_ts=ts, months=6)
        assert sigma > 0.0

    @skip_if_no_data
    def test_reasonable_magnitude(self, btc_df):
        # BTC hourly vol is typically 0.3%–1.5%
        ts = pd.Timestamp("2026-01-13 17:00:00", tz="UTC")
        sigma = _calibrate_sigma(btc_df, before_ts=ts, months=6)
        assert 0.001 < sigma < 0.05

    @skip_if_no_data
    def test_insufficient_data_raises(self, btc_df):
        # Ask for 6 months before the very first row — window is empty
        ts = pd.Timestamp("2012-01-03 00:00:00", tz="UTC")
        with pytest.raises(ValueError, match="Insufficient"):
            _calibrate_sigma(btc_df, before_ts=ts, months=6)


# ---------------------------------------------------------------------------
# Integration tests: build_synthetic_ohlcv
# ---------------------------------------------------------------------------

class TestBuildSyntheticOhlcv:
    @skip_if_no_data
    def test_output_columns(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]

    @skip_if_no_data
    def test_prices_within_bounds(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        assert (df["close"] >= PRICE_FLOOR).all()
        assert (df["close"] <= PRICE_CEIL).all()

    @skip_if_no_data
    def test_final_candle_is_settlement(self, btc_df, jan20_contracts):
        # $90k YES settled YES → final close = PRICE_CEIL
        c90 = next(x for x in jan20_contracts if x.strike == 90_000 and x.direction == "above")
        df90 = build_synthetic_ohlcv(btc_df, c90, sigma_1h=0.01)
        assert df90["close"].iloc[-1] == pytest.approx(PRICE_CEIL)

        # $96k YES settled NO → final close = PRICE_FLOOR
        c96 = next(x for x in jan20_contracts if x.strike == 96_000 and x.direction == "above")
        df96 = build_synthetic_ohlcv(btc_df, c96, sigma_1h=0.01)
        assert df96["close"].iloc[-1] == pytest.approx(PRICE_FLOOR)

    @skip_if_no_data
    def test_row_count_matches_btc_window(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        # Jan 13 17:00 → Jan 20 17:00 = 168 hours → 169 candles (inclusive)
        assert len(df) >= 168

    @skip_if_no_data
    def test_high_ge_low(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 92_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        assert (df["high"] >= df["low"]).all()

    @skip_if_no_data
    def test_date_is_integer_ms(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        assert df["date"].dtype == np.dtype("int64")
        # Sanity: dates should be in milliseconds (order of 1e12)
        assert df["date"].iloc[0] > 1_000_000_000_000

    @skip_if_no_data
    def test_reproducibility(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df1 = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01, random_seed=7)
        df2 = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01, random_seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    @skip_if_no_data
    def test_different_seeds_differ(self, btc_df, jan20_contracts):
        c = next(x for x in jan20_contracts if x.strike == 90_000)
        df1 = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01, random_seed=1)
        df2 = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01, random_seed=2)
        # Interior candles differ; final settlement candle is deterministic
        assert not (df1["close"].iloc[:-1] == df2["close"].iloc[:-1]).all()

    @skip_if_no_data
    def test_nontrivial_price_path(self, btc_df, jan20_contracts):
        # The path should not be constant (stuck at 0 or 1 the whole time)
        c = next(x for x in jan20_contracts if x.strike == 92_000)
        df = build_synthetic_ohlcv(btc_df, c, sigma_1h=0.01)
        interior = df["close"].iloc[:-1]
        assert interior.std() > 0.005, "Price path appears degenerate (too little variation)"
        assert interior.min() > 0.01, "Price path collapsed near 0 prematurely"
        assert interior.max() < 0.99, "Price path collapsed near 1 prematurely"

"""Tests for adaptive_trend strategy."""

# to test this file, run the command on bash: python -m pytest tests/test_adaptive_trend_strategy.py -q

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from strategy.adaptive_trend.adaptive_trend import adaptive_trend

def _make_ohlcv(n: int = 300, start: str = "2025-01-01") -> pd.DataFrame:
    """Create synthetic 4h OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n, freq="4H", tz="UTC")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(100, 10000, size=n).astype(float),
        }
    )


def _make_market_cap_csvs(base_dir: Path):
    """Create fake market-cap CSV files beside the strategy file."""
    snapped = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")

    btc = pd.DataFrame(
        {
            "snapped_at": snapped,
            "market_cap": np.linspace(1_000_000_000, 1_200_000_000, len(snapped)),
        }
    )
    eth = pd.DataFrame(
        {
            "snapped_at": snapped,
            "market_cap": np.linspace(800_000_000, 900_000_000, len(snapped)),
        }
    )
    xrp = pd.DataFrame(
        {
            "snapped_at": snapped,
            "market_cap": np.linspace(200_000_000, 250_000_000, len(snapped)),
        }
    )

    btc.to_csv(base_dir / "btc-usd-max.csv", index=False)
    eth.to_csv(base_dir / "eth-usd-max.csv", index=False)
    xrp.to_csv(base_dir / "xrp-usd-max.csv", index=False)


class DummyDP:
    """Minimal dataprovider stub for custom_stoploss."""

    def __init__(self, analyzed_df: pd.DataFrame):
        self.analyzed_df = analyzed_df

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return self.analyzed_df, None


class DummyTrade:
    """Minimal trade stub for stoploss / exit tests."""

    def __init__(self, is_short: bool = False, open_date_utc: datetime | None = None):
        self.is_short = is_short
        self.user_data = {}
        self.open_date_utc = open_date_utc or datetime.now(timezone.utc)

    def get_custom_data(self, key: str):
        return self.user_data.get(key)

    def set_custom_data(self, key: str, val):
        self.user_data[key] = val


class TestAdaptiveTrendHelpers:
    def test_base_symbol_extracts_base(self):
        strat = adaptive_trend({})
        assert strat._base_symbol("ETH/USDT") == "ETH"
        assert strat._base_symbol("btc/usdc") == "BTC"


class TestAdaptiveTrendMarketCap:
    def test_load_market_cap_data_returns_dataframe(self, monkeypatch, tmp_path):
        strat = adaptive_trend({})
        fake_strategy_file = tmp_path / "adaptive_trend.py"
        fake_strategy_file.write_text("# dummy")

        _make_market_cap_csvs(tmp_path)

        monkeypatch.setattr(adaptive_trend, "__module__", adaptive_trend.__module__)
        monkeypatch.setattr(
            Path,
            "resolve",
            lambda self: fake_strategy_file,
        )

        result = strat.load_market_cap_data()

        assert result is not None
        assert not result.empty
        for col in ["date", "symbol", "marketCap", "rank", "total_count"]:
            assert col in result.columns

    def test_merge_market_cap_adds_expected_columns(self):
        strat = adaptive_trend({})
        df = _make_ohlcv(50)

        mcap = pd.DataFrame(
            {
                "date": pd.to_datetime(df["date"], utc=True).dt.floor("D").unique().repeat(3),
                "symbol": ["BTC", "ETH", "XRP"] * len(pd.to_datetime(df["date"], utc=True).dt.floor("D").unique()),
                "marketCap": [1000, 800, 200] * len(pd.to_datetime(df["date"], utc=True).dt.floor("D").unique()),
            }
        )
        mcap["rank"] = mcap.groupby("date")["marketCap"].rank(ascending=False, method="min")
        mcap["total_count"] = mcap.groupby("date")["symbol"].transform("count")

        strat._mcap_df = mcap
        result = strat._merge_market_cap(df.copy(), "BTC/USDT")

        for col in ["mcap_rank", "mcap_total", "allow_long_mcap", "allow_short_mcap"]:
            assert col in result.columns

        assert set(result["allow_long_mcap"].unique()).issubset({0, 1})
        assert set(result["allow_short_mcap"].unique()).issubset({0, 1})

    def test_merge_market_cap_defaults_to_allow_when_no_data(self):
        strat = adaptive_trend({})
        strat._mcap_df = None

        # Force loader to return None
        strat.load_market_cap_data = lambda: None

        df = _make_ohlcv(20)
        result = strat._merge_market_cap(df.copy(), "BTC/USDT")

        assert (result["allow_long_mcap"] == 1).all()
        assert (result["allow_short_mcap"] == 1).all()


class TestAdaptiveTrendIndicators:
    def test_populate_indicators_adds_expected_columns(self, monkeypatch):
        strat = adaptive_trend({})
        strat._mcap_df = None
        strat.load_market_cap_data = lambda: None

        df = _make_ohlcv(300)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        expected_cols = [
            "mom",
            "atr",
            "ret",
            "sr_long",
            "sr_short",
            "mcap_rank",
            "mcap_total",
            "allow_long_mcap",
            "allow_short_mcap",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_momentum_has_nans_before_lookback(self):
        strat = adaptive_trend({})
        strat._mcap_df = None
        strat.load_market_cap_data = lambda: None

        df = _make_ohlcv(300)
        L = int(strat.mom_lookback.value)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        assert result["mom"].iloc[:L].isna().all()

    def test_atr_is_numeric_after_warmup(self):
        strat = adaptive_trend({})
        strat._mcap_df = None
        strat.load_market_cap_data = lambda: None

        df = _make_ohlcv(300)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        valid = result["atr"].dropna()
        assert not valid.empty
        assert valid.dtype in [np.float64, np.float32]

    def test_sharpe_columns_are_numeric(self):
        strat = adaptive_trend({})
        strat._mcap_df = None
        strat.load_market_cap_data = lambda: None

        df = _make_ohlcv(300)
        result = strat.populate_indicators(df.copy(), metadata={"pair": "BTC/USDT"})

        assert result["sr_long"].dtype in [np.float64, np.float32]
        assert result["sr_short"].dtype in [np.float64, np.float32]


class TestAdaptiveTrendEntries:
    def test_populate_entry_trend_sets_enter_long(self):
        strat = adaptive_trend({})

        df = pd.DataFrame(
            {
                "volume": [1000, 1000, 1000],
                "mom": [0.01, 0.05, 0.10],
                "sr_long": [0.1, 0.5, 1.0],
                "sr_short": [0.1, 0.1, 0.1],
                "allow_long_mcap": [1, 1, 1],
                "allow_short_mcap": [0, 0, 0],
            }
        )

        result = strat.populate_entry_trend(df.copy(), metadata={"pair": "BTC/USDT"})
        assert "enter_long" in result.columns
        assert result["enter_long"].fillna(0).sum() >= 1

    def test_populate_entry_trend_sets_enter_short_when_conditions_met(self):
        strat = adaptive_trend({})

        df = pd.DataFrame(
            {
                "volume": [1000, 1000, 1000],
                "mom": [-0.01, -0.10, -0.20],
                "sr_long": [0.1, 0.1, 0.1],
                "sr_short": [0.5, 2.0, 2.5],
                "allow_long_mcap": [0, 0, 0],
                "allow_short_mcap": [1, 1, 1],
            }
        )

        result = strat.populate_entry_trend(df.copy(), metadata={"pair": "BTC/USDT"})
        assert "enter_short" in result.columns
        assert result["enter_short"].fillna(0).sum() >= 1

    def test_no_entry_when_volume_zero(self):
        strat = adaptive_trend({})

        df = pd.DataFrame(
            {
                "volume": [0, 0],
                "mom": [0.10, -0.10],
                "sr_long": [10, 10],
                "sr_short": [10, 10],
                "allow_long_mcap": [1, 1],
                "allow_short_mcap": [1, 1],
            }
        )

        result = strat.populate_entry_trend(df.copy(), metadata={"pair": "BTC/USDT"})
        assert result.get("enter_long", pd.Series([0, 0])).fillna(0).sum() == 0
        assert result.get("enter_short", pd.Series([0, 0])).fillna(0).sum() == 0


class TestAdaptiveTrendExits:
    def test_populate_exit_trend_sets_zero_exits(self):
        strat = adaptive_trend({})
        df = _make_ohlcv(20)

        result = strat.populate_exit_trend(df.copy(), metadata={"pair": "BTC/USDT"})

        assert "exit_long" in result.columns
        assert "exit_short" in result.columns
        assert (result["exit_long"] == 0).all()
        assert (result["exit_short"] == 0).all()

    def test_custom_exit_returns_time_exit_after_hold_days(self):
        strat = adaptive_trend({})
        trade = DummyTrade(open_date_utc=datetime.now(timezone.utc) - timedelta(days=61))

        result = strat.custom_exit(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.02,
        )

        assert result == "time_exit"

    def test_custom_exit_returns_none_before_hold_days(self):
        strat = adaptive_trend({})
        trade = DummyTrade(open_date_utc=datetime.now(timezone.utc) - timedelta(days=10))

        result = strat.custom_exit(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.02,
        )

        assert result is None


class TestAdaptiveTrendCustomStoploss:
    def test_custom_stoploss_returns_default_when_no_dataframe(self):
        strat = adaptive_trend({})
        strat.dp = DummyDP(pd.DataFrame())

        trade = DummyTrade(is_short=False)
        result = strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )

        assert result == 1

    def test_custom_stoploss_long_sets_trailing_stop(self):
        strat = adaptive_trend({})

        analyzed = pd.DataFrame(
            {
                "atr": [1.5, 2.0],
            }
        )
        strat.dp = DummyDP(analyzed)

        trade = DummyTrade(is_short=False)
        result = strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )

        assert -0.99 <= result <= 0.0
        assert "atr_trail_stop" in trade.user_data
        assert trade.user_data["atr_trail_stop"] < 100.0

    def test_custom_stoploss_short_sets_trailing_stop(self):
        strat = adaptive_trend({})

        analyzed = pd.DataFrame(
            {
                "atr": [1.5, 2.0],
            }
        )
        strat.dp = DummyDP(analyzed)

        trade = DummyTrade(is_short=True)
        result = strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )

        assert -0.99 <= result <= 0.0
        assert "atr_trail_stop" in trade.user_data
        assert trade.user_data["atr_trail_stop"] > 100.0

    def test_custom_stoploss_long_only_trails_up(self):
        strat = adaptive_trend({})
        strat.dp = DummyDP(pd.DataFrame({"atr": [2.0]}))

        trade = DummyTrade(is_short=False)

        first = strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            current_profit=0.01,
        )
        first_stop = trade.user_data["atr_trail_stop"]

        second = strat.custom_stoploss(
            pair="BTC/USDT",
            trade=trade,
            current_time=datetime.now(timezone.utc),
            current_rate=95.0,
            current_profit=-0.03,
        )
        second_stop = trade.user_data["atr_trail_stop"]

        assert second_stop >= first_stop
        assert -0.99 <= first <= 0.0
        assert -0.99 <= second <= 0.0


class TestAdaptiveTrendStakeSizing:
    def test_custom_stake_amount_long_uses_proposed_stake(self):
        strat = adaptive_trend({})

        stake = strat.custom_stake_amount(
            pair="BTC/USDT",
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            proposed_stake=1000.0,
            min_stake=100.0,
            max_stake=5000.0,
            leverage=1.0,
            side="long",
        )

        assert 100.0 <= stake <= 5000.0
        assert stake == 1000.0

    def test_custom_stake_amount_short_scales_down(self):
        strat = adaptive_trend({})

        stake = strat.custom_stake_amount(
            pair="BTC/USDT",
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            proposed_stake=1000.0,
            min_stake=100.0,
            max_stake=5000.0,
            leverage=1.0,
            side="short",
        )

        assert 100.0 <= stake <= 5000.0
        assert stake < 1000.0

    def test_custom_stake_amount_clamps_to_bounds(self):
        strat = adaptive_trend({})

        stake = strat.custom_stake_amount(
            pair="BTC/USDT",
            current_time=datetime.now(timezone.utc),
            current_rate=100.0,
            proposed_stake=50.0,
            min_stake=100.0,
            max_stake=5000.0,
            leverage=1.0,
            side="long",
        )

        assert stake == 100.0
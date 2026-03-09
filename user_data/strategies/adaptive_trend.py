# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Optional
from pandas import DataFrame
# --------------------------------

from freqtrade.strategy import DecimalParameter, IntParameter
import talib.abstract as ta
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta


class adaptive_trend(IStrategy):
    """
    AdaptiveTrend-inspired strategy (Nguyen 2026) implemented for Freqtrade.

    Key modules implemented:
      - H6 momentum entry signal
      - ATR-multiplier dynamic trailing stop (custom_stoploss)
      - Rolling Sharpe filter (proxy for monthly SR selection)
      - Market-cap aware filter (top-K long, bottom-K short) via offline CSV
      - 70/30 long-short stake scaling (approximation)

    Breakdown of how the strategy works
    1. Start with broad universe of tradable crypto pairs and collect 6hr OHLCV candles 
    2. At the start of each month, rebuild the tradeable universe
    3. Apply market-cap filter to crypto-pairs
    4. apply performance filter using recent sharpe ratio, coins with strong sharpe are eligible
    5. for each eligible asset, compute momentum based on returns
    6. Compute ATR over k periods. after long, initialize a trailing stop based on S = Pt - b x ATRt
    7. After selecting which assets qualify, allocate capital asymmetrically: 
    roughly 70% to longs and 30% to shorts.
    This reflects the paper’s view that crypto has a long-run positive drift, 
    so the portfolio should not be perfectly market-neutral. 
    8. At the next monthly rebalance, repeat the universe-selection process again: 
    re-check market-cap eligibility, re-check Sharpe, and allow coins to enter 
    or leave the tradable set. 

    """

    INTERFACE_VERSION: int = 3

    # Paper uses 6-hour candles. (H6)
    timeframe = "6h"
    can_short = False

    # Let custom_stoploss manage exits. Keep a hard stop as safety.
    stoploss = -0.99
    minimal_roi = {}

    # Startup candles: need enough for MOM(L), ATR(k), and Sharpe window.
    # Default Sharpe window below is 30 days ≈ 30 * 4 = 120 candles.
    startup_candle_count = 250

    # Optional order type mapping (you can change to market if you want).
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Hyperoptable parameters
    # Momentum lookback L in candles (paper uses L as a tunable lookback).
    mom_lookback = IntParameter(8, 80, default=24, space="buy")  # 24 * 6h = 6 days

    # Entry threshold θ_entry
    theta_entry = DecimalParameter(0.002, 0.08, decimals=3, default=0.05, space="buy")

    # ATR period k and multiplier α for trailing stop
    atr_period = IntParameter(7, 40, default=14, space="sell")
    atr_mult = DecimalParameter(1.5, 4.5, decimals=2, default=2.50, space="sell")

    # Rolling Sharpe window (proxy for “previous month”)
    sharpe_window_candles = IntParameter(80, 200, default=120, space="buy")  # ~30d

    # Sharpe thresholds γ_L, γ_S (paper uses 1.3 / 1.7)
    gamma_long = DecimalParameter(0.5, 3.0, decimals=2, default=1.30, space="buy")
    gamma_short = DecimalParameter(0.5, 3.5, decimals=2, default=1.70, space="buy")

    # Market-cap filter sizes (paper uses KL=15 and bottom-KS).
    top_k_long = IntParameter(5, 30, default=10, space="buy")
    bottom_k_short = IntParameter(5, 60, default=30, space="buy")

    # Asymmetric allocation λ = 0.7 (long) / 0.3 (short).
    long_alloc = DecimalParameter(0.50, 0.90, decimals=2, default=0.70, space="buy")

    # Market-cap data handling
    _mcap_df: Optional[pd.DataFrame] = None

    def _base_symbol(self, pair: str) -> str:
        # "ETH/USDT" -> "ETH"
        return pair.split("/")[0].strip().upper()

    def load_market_cap_data(self) -> Optional[pd.DataFrame]:
        """
        Expected CSV (offline) format (example):
          date,symbol,marketCap
          2024-01-01,ETH,250000000000
          2024-01-01,SOL,45000000000
          ...

        We compute per-date rank + total_count internally.
        """
        if self._mcap_df is not None:
            return self._mcap_df

        base_dir = Path(__file__).resolve().parent
        path = base_dir / "coingecko_marketcap_daily.csv"  # YOU provide this file
        if not path.exists():
            # If absent, we simply disable market-cap filtering.
            self._mcap_df = None
            return None

        df = pd.read_csv(path)
        if not {"date", "symbol", "marketCap"}.issubset(df.columns):
            raise ValueError("Market cap CSV must contain: date,symbol,marketCap")

        df["date"] = pd.to_datetime(df["date"], utc=True).dt.floor("D")
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")

        # Rank per date (1 = largest cap)
        df["rank"] = df.groupby("date")["marketCap"].rank(ascending=False, method="min")
        df["total_count"] = df.groupby("date")["symbol"].transform("count")

        self._mcap_df = df
        return df

    def _merge_market_cap(self, dataframe: DataFrame, pair: str) -> DataFrame:
        mcap = self.load_market_cap_data()
        dataframe["mcap_rank"] = np.nan
        dataframe["mcap_total"] = np.nan

        if mcap is None:
            dataframe["allow_long_mcap"] = 1
            dataframe["allow_short_mcap"] = 1
            return dataframe

        sym = self._base_symbol(pair)

        # Candle date in UTC floored to day to match mcap df.
        dataframe["date_day"] = pd.to_datetime(dataframe["date"], utc=True).dt.floor("D")

        mcap_sym = mcap[mcap["symbol"] == sym][["date", "rank", "total_count"]].copy()
        mcap_sym = mcap_sym.rename(columns={"date": "date_day", "rank": "mcap_rank", "total_count": "mcap_total"})

        dataframe = dataframe.merge(mcap_sym, on="date_day", how="left")

        # If no cap info for that day, be conservative: disallow both.
        dataframe["allow_long_mcap"] = (
            (dataframe["mcap_rank"].notna())
            & (dataframe["mcap_rank"] <= float(self.top_k_long.value))
        ).astype(int)

        dataframe["allow_short_mcap"] = (
            (dataframe["mcap_rank"].notna())
            & (dataframe["mcap_total"].notna())
            & (dataframe["mcap_rank"] >= (dataframe["mcap_total"] - float(self.bottom_k_short.value) + 1))
        ).astype(int)

        return dataframe

    # Core indicators
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")

        # Momentum MOM_t = (P_t - P_{t-L}) / P_{t-L}
        L = int(self.mom_lookback.value)
        dataframe["mom"] = (dataframe["close"] - dataframe["close"].shift(L)) / dataframe["close"].shift(L)

        # ATR over k periods
        k = int(self.atr_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=k)

        # Returns (for Sharpe proxy)
        dataframe["ret"] = dataframe["close"].pct_change()

        # Rolling Sharpe (annualized) on H6 frequency: ~1460 bars/year (365*4)
        w = int(self.sharpe_window_candles.value)
        eps = 1e-12
        mean_r = dataframe["ret"].rolling(w).mean()
        std_r = dataframe["ret"].rolling(w).std().replace(0, np.nan)
        ann_factor = np.sqrt(365.0 * 4.0)

        dataframe["sr_long"] = (mean_r / (std_r + eps)) * ann_factor
        # “short Sharpe” = Sharpe of (-ret) (i.e., profit when price falls)
        dataframe["sr_short"] = ((-mean_r) / (std_r + eps)) * ann_factor

        # Market cap filter (top/bottom K)
        dataframe = self._merge_market_cap(dataframe, pair)

        # Clean up early NaNs (don’t invent signals before indicators exist)
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

        return dataframe

    # Entries
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        theta = float(self.theta_entry.value)
        gL = float(self.gamma_long.value)
        gS = float(self.gamma_short.value)

        # Long: MOM > θ and SR filter passes and market-cap filter passes
        dataframe.loc[
            (
                (dataframe["volume"] > 0)
                & (dataframe["mom"] > theta)
                & (dataframe["sr_long"] >= gL)
                & (dataframe["allow_long_mcap"] == 1)
            ),
            "enter_long"
        ] = 1

        # Short: MOM < -θ and SR filter passes and market-cap filter passes
        dataframe.loc[
            (
                (dataframe["volume"] > 0)
                & (dataframe["mom"] < -theta)
                & (dataframe["sr_short"] >= gS)
                & (dataframe["allow_short_mcap"] == 1)
            ),
            "enter_short"
        ] = 1

        return dataframe

    # Let trailing stop manage exits (so we don't set exit signals here).
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    # Dynamic trailing stop (ATR-based)
    def custom_stoploss(
        self,
        pair: str,
        trade,
        current_time,
        current_rate,
        current_profit,
        **kwargs
    ) -> float:
        """
        Paper-style dynamic trailing stop:
          For long: S_t = max(S_{t-1}, P_t - α*ATR_t)
          For short: S_t = min(S_{t-1}, P_t + α*ATR_t)

        We store the stop price in trade custom data so it “trails”.
        Returns stoploss as a negative value (relative stop) as required by Freqtrade.
        """
        df = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return 1  # keep default / do nothing

        last = df.iloc[-1]
        atr = last.get("atr", None)
        if atr is None or np.isnan(atr) or atr <= 0:
            return 1

        alpha = float(self.atr_mult.value)

        # Helpers for cross-version compatibility
        def _get(key: str):
            if hasattr(trade, "get_custom_data"):
                return trade.get_custom_data(key)
            if hasattr(trade, "user_data") and isinstance(trade.user_data, dict):
                return trade.user_data.get(key)
            return None

        def _set(key: str, val):
            if hasattr(trade, "set_custom_data"):
                trade.set_custom_data(key, val)
                return
            if hasattr(trade, "user_data"):
                if trade.user_data is None:
                    trade.user_data = {}
                trade.user_data[key] = val

        prev_stop = _get("atr_trail_stop")

        is_short = getattr(trade, "is_short", False)

        if not is_short:
            # Long trailing stop price
            candidate = current_rate - alpha * atr
            if prev_stop is None:
                new_stop = candidate
            else:
                new_stop = max(float(prev_stop), candidate)
        else:
            # Short trailing stop price
            candidate = current_rate + alpha * atr
            if prev_stop is None:
                new_stop = candidate
            else:
                new_stop = min(float(prev_stop), candidate)

        _set("atr_trail_stop", float(new_stop))

        # Convert stop price into a relative stoploss (negative float)
        if not is_short:
            rel = (new_stop / current_rate) - 1.0
        else:
            # For shorts: stop triggers when price rises above stop.
            # Relative "loss" if price moves from current_rate to stop.
            rel = 1.0 - (new_stop / current_rate)

        # Clamp: Freqtrade expects >= -1 and <= 0 for stoploss.
        rel = float(np.clip(rel, -0.99, 0.0))
        return rel

    # Asymmetric 70/30 allocation (approx)
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str] = None,
        side: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Paper uses λ=0.7 long / 0.3 short.
        True equal-weight across each leg requires portfolio-level coordination.
        Here we approximate by scaling stake per trade based on side.
        """
        lam = float(self.long_alloc.value)
        lam = float(np.clip(lam, 0.01, 0.99))
        # Relative short vs long sizing
        short_scale = (1.0 - lam) / lam

        stake = proposed_stake
        if side is not None and side.lower() == "short":
            stake = proposed_stake * short_scale

        return float(np.clip(stake, min_stake, max_stake))

    # Optional: time-based safety exit (paper’s system is stop-driven; keep if you want)
    HOLD_DAYS = 60

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # prevent zombie positions
        if current_time - trade.open_date_utc >= timedelta(days=self.HOLD_DAYS):
            return "time_exit"
        return None
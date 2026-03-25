# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class DonchianChannelStrategy(IStrategy):
    """
    Donchian Channel Trend Following Strategy (adapted from Curtis Faith's The Way Of The Turtle)

    Rules implemented:
      - Entry: 20-period Donchian breakout.
      - Trend filter: SMA(15) vs SMA(350)
          * if SMA15 > SMA350 => allow longs
          * if SMA15 < SMA350 => allow shorts (optional)
      - Exit: 10-period Donchian exit OR time exit after 80 candles (eg: 80 days on 1d timeframe)

    Note:
      - Use timeframe = "1d" to match “20-day / 10-day / 80-day” wording.
      - Donchian levels are shifted by 1 candle to avoid lookahead.
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    minimal_roi = {}
    stoploss = -0.20
    trailing_stop = False

    timeframe = "1d"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 400  # SMA350 + channel windows

    # Period parameters
    entry_dc_period = 20
    exit_dc_period = 10
    fast_ma_period = 15
    slow_ma_period = 350

    # Time-exit in candles (80 days if timeframe == "1d")
    time_exit_candles = 80

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ma_fast"] = ta.SMA(dataframe, timeperiod=self.fast_ma_period)
        dataframe["ma_slow"] = ta.SMA(dataframe, timeperiod=self.slow_ma_period)

        # Donchian channels - shift(1) prevents using the current candle in the threshold.
        dataframe["dc_upper_entry"] = dataframe["high"].rolling(self.entry_dc_period).max().shift(1)
        dataframe["dc_lower_entry"] = dataframe["low"].rolling(self.entry_dc_period).min().shift(1)

        dataframe["dc_upper_exit"] = dataframe["high"].rolling(self.exit_dc_period).max().shift(1)
        dataframe["dc_lower_exit"] = dataframe["low"].rolling(self.exit_dc_period).min().shift(1)

        dataframe["trend_up"] = dataframe["ma_fast"] > dataframe["ma_slow"]
        dataframe["trend_down"] = dataframe["ma_fast"] < dataframe["ma_slow"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Long breakout entry
        dataframe.loc[
            (
                (dataframe["trend_up"]) &
                (dataframe["close"] > dataframe["dc_upper_entry"]) &
                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        # Short breakout entry (only if enabled)
        dataframe.loc[
            (
                (self.can_short) &
                (dataframe["trend_down"]) &
                (dataframe["close"] < dataframe["dc_lower_entry"]) &
                (dataframe["volume"] > 0)
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Donchian exit for longs: break below 10-period low
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["dc_lower_exit"])
            ),
            "exit_long",
        ] = 1

        # Donchian exit for shorts: break above 10-period high
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["dc_upper_exit"])
            ),
            "exit_short",
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        Time-based exit: close trade after N candles. (1d timeframe = N days)
        Convert candle-count to minutes using timeframe_to_minutes(self.timeframe).
        """
        tf_min = timeframe_to_minutes(self.timeframe)
        max_age_min = self.time_exit_candles * tf_min

        # Ensure timezone-aware datetimes
        open_dt = trade.open_date_utc # gives trade's open date
        if open_dt.tzinfo is None:
            open_dt = open_dt.replace(tzinfo=timezone.utc)

        now_dt = current_time
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        age_minutes = (now_dt - open_dt).total_seconds() / 60.0

        if age_minutes >= max_age_min:
            return f"time_exit_{self.time_exit_candles}c"

        return None
    
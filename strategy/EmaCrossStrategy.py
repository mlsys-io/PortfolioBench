# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple

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
    AnnotationType,
)

from alpha.SimpleEmaFactors import EmaAlpha

# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class EmaCrossStrategy(IStrategy):
    """
    Strategy adapted from paper: http://arxiv.org/abs/2511.00665 
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30

    ema_fast_length = IntParameter(5, 15, default=12, space="buy")
    ema_slow_length = IntParameter(20, 30, default=26, space="buy")
    ema_exit_length = IntParameter(5, 10, default=6, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_length.value)
        # dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_length.value)
        # dataframe["ema_exit"] = ta.EMA(dataframe, timeperiod=self.ema_exit_length.value)
        # dataframe["mean-volume"] = dataframe["volume"].rolling(20).mean()
        dataframe = EmaAlpha(dataframe, metadata).process()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe["ema_fast"], dataframe["ema_slow"]))
                & (dataframe["mean-volume"] > 0.75)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_below(dataframe["ema_exit"], dataframe["ema_fast"])), "exit_long"
        ] = 1
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_close = dataframe.iloc[-1]["close"]
        max_deviation = 0.01  # 1% deviation allowed
        deviation = abs(rate - last_close) / last_close

        if deviation > max_deviation:
            return False

        return True
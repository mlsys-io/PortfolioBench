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

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class StochasticCciStrategy(IStrategy):
    """
    Stochastic Oscillator + Commodity Channel Index (CCI) momentum strategy.

    Combines the Stochastic Oscillator (Lane, 1984) with the CCI (Lambert, 1980)
    for dual momentum confirmation. Enters when both indicators signal oversold
    conditions in an uptrend; exits on overbought or trend reversal.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"

    can_short: bool = False

    minimal_roi = {}

    stoploss = -99

    trailing_stop = False

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    # Strategy parameters
    stoch_k_period = IntParameter(10, 20, default=14, space="buy")
    stoch_d_period = IntParameter(3, 5, default=3, space="buy")
    stoch_slowing = IntParameter(3, 5, default=3, space="buy")
    cci_period = IntParameter(14, 30, default=20, space="buy")
    sma_period = IntParameter(150, 250, default=200, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(
            dataframe,
            fastk_period=self.stoch_k_period.value,
            slowk_period=self.stoch_d_period.value,
            slowd_period=self.stoch_slowing.value,
        )
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]

        dataframe["cci"] = ta.CCI(dataframe, timeperiod=self.cci_period.value)

        dataframe["sma200"] = ta.SMA(dataframe, timeperiod=self.sma_period.value)

        dataframe["mean-volume"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Stochastic %K crosses above %D in oversold zone
                (qtpylib.crossed_above(dataframe["slowk"], dataframe["slowd"]))
                & (dataframe["slowk"] < 30)
                # CCI confirms oversold bounce
                & (dataframe["cci"] > -100)
                & (dataframe["cci"].shift(1) <= -100)
                # Price above long-term SMA (uptrend filter)
                & (dataframe["close"] > dataframe["sma200"])
                & (dataframe["mean-volume"] > 0.75)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Stochastic overbought
                (
                    (dataframe["slowk"] > 80)
                    & (qtpylib.crossed_below(dataframe["slowk"], dataframe["slowd"]))
                )
                # OR CCI overbought reversal
                | (
                    (dataframe["cci"] < 100)
                    & (dataframe["cci"].shift(1) >= 100)
                )
            ),
            "exit_long",
        ] = 1
        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: str | None,
        side: str, **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_close = dataframe.iloc[-1]["close"]
        max_deviation = 0.01
        deviation = abs(rate - last_close) / last_close
        if deviation > max_deviation:
            return False
        return True

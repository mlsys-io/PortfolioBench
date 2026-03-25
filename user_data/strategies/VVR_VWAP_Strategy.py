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

import talib.abstract as ta
from technical import qtpylib

class VVR_VWAP_Strategy(IStrategy):
    """
    Strategy adapted from paper: https://arxiv.org/pdf/2508.01419
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
    stoploss = -0.02

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 1. EMAs for Trend
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=26)

        # 2. RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # 3. MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        # 4. VVR (Volume-to-Volatility Ratio)
        epsilon = 1e-6
        dataframe["price_range"] = dataframe["high"] - dataframe["low"]
        # Use rolling mean of VVR immediately for comparison
        dataframe["vvr"] = dataframe["volume"] / (dataframe["price_range"] + epsilon)
        dataframe["vvr_mean"] = dataframe["vvr"].rolling(window=50).mean()

        # 5. Rolling VWAP (approx 1 day on 5m candles = 288 candles)
        rolling_window = 288
        vwap_num = (dataframe['volume'] * (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3).rolling(window=rolling_window).sum()
        vwap_denom = dataframe['volume'].rolling(window=rolling_window).sum()
        dataframe['vwap'] = vwap_num / vwap_denom

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # 1. TREND: Long term trend is UP
                (dataframe["ema_fast"] > dataframe["ema_slow"]) &

                # 2. MOMENTUM: RSI is not Overbought (Safe to buy) 
                (dataframe["rsi"] < 55) &
                (dataframe["rsi"] > 30) & 

                # 3. MACD: Momentum is recovering or positive
                (dataframe["macd"] > dataframe["macdsignal"]) &

                # 4. VALUATION: Price is below or near the daily VWAP
                (dataframe["close"] < dataframe["vwap"]) &

                # 5. LIQUIDITY: High efficiency (Volume > Volatility)
                (dataframe["vvr"] > dataframe["vvr_mean"]) &

                (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Exit if RSI gets too hot
                (dataframe["rsi"] > 75) |

                # OR MACD crosses down
                (qtpylib.crossed_below(dataframe["macd"], dataframe["macdsignal"])) |

                # OR Price extends too far above VWAP
                (dataframe["close"] > dataframe["vwap"] * 1.03)
            ),
            "exit_long",
        ] = 1

        return dataframe
    
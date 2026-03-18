"""
ZScoreMeanReversionStrategy — Statistical mean-reversion on Bollinger Z-score
==============================================================================
Logic:
  Entry  : Z-score of close relative to BB drops below -1.5 (oversold)
            AND RSI < 35 AND volume spike (>1.5× 20-period average)
            AND price > long-term EMA200 (only buy dips in uptrend)
  Exit   : Z-score returns to 0 (mean) OR RSI > 65
  Stop   : Fixed 5%

Suitable for: crypto / US stocks, 4h or 1d timeframe
"""

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np


class ZScoreMeanReversionStrategy(IStrategy):
    """AI-generated mean-reversion strategy using Bollinger Band Z-score."""

    timeframe = "4h"
    minimal_roi = {"0": 0.15, "720": 0.08, "2160": 0.03}
    stoploss = -0.05
    trailing_stop = False
    can_short = False
    startup_candle_count = 210

    # Hyperopt-ready parameters
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space="buy")
    zscore_entry = DecimalParameter(-2.5, -1.0, default=-1.5, space="buy")
    zscore_exit = DecimalParameter(-0.3, 0.5, default=0.0, space="sell")
    rsi_entry = IntParameter(25, 45, default=35, space="buy")
    rsi_exit = IntParameter(55, 75, default=65, space="sell")
    volume_mult = DecimalParameter(1.2, 2.5, default=1.5, space="buy")
    trend_ema = IntParameter(150, 250, default=200, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bb = ta.BBANDS(
            dataframe,
            timeperiod=self.bb_period.value,
            nbdevup=self.bb_std.value,
            nbdevdn=self.bb_std.value,
        )
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]

        # Z-score: how many std-devs is close from the BB mid?
        bb_std_val = (dataframe["bb_upper"] - dataframe["bb_mid"]) / self.bb_std.value
        dataframe["zscore"] = (dataframe["close"] - dataframe["bb_mid"]) / bb_std_val.replace(0, np.nan)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Volume filter: rolling 20-bar average
        dataframe["vol_ma"] = dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["vol_ma"].replace(0, np.nan)

        # Long-term trend EMA
        dataframe["ema_trend"] = ta.EMA(dataframe, timeperiod=self.trend_ema.value)

        # Bandwidth (squeeze detection — avoid trading in low-vol compression)
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["zscore"] < self.zscore_entry.value)  # statistically oversold
                & (dataframe["rsi"] < self.rsi_entry.value)  # momentum confirms weakness
                & (dataframe["vol_ratio"] > self.volume_mult.value)  # volume spike (capitulation)
                & (dataframe["close"] > dataframe["ema_trend"])  # in long-term uptrend
                & (dataframe["bb_width"] > 0.02)  # not in extreme squeeze
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["zscore"] > self.zscore_exit.value)  # mean-reverted
                | (dataframe["rsi"] > self.rsi_exit.value)  # overbought
                | (dataframe["close"] < dataframe["bb_lower"])  # breakdown (stop cascade)
            ),
            "exit_long",
        ] = 1
        return dataframe

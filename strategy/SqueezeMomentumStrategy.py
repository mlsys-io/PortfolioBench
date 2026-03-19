"""
SqueezeMomentumStrategy — Volatility squeeze + momentum breakout
================================================================
Logic:
  Detects "squeeze" when Bollinger Bands are inside Keltner Channels
  (low volatility compression). Enters when squeeze releases upward
  with positive momentum.

  Entry  : Squeeze just released (BB outside KC) AND
            momentum histogram positive and rising AND
            RSI > 50
  Exit   : Momentum turns negative OR RSI > 75
  Stop   : 5%
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np


class SqueezeMomentumStrategy(IStrategy):
    """Squeeze momentum breakout strategy (LazyBear-inspired)."""

    timeframe = "5m"
    minimal_roi = {"0": 0.18, "180": 0.09, "540": 0.04}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.025
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    can_short = False
    startup_candle_count = 30

    bb_period = IntParameter(15, 25, default=20, space="buy")
    kc_period = IntParameter(15, 25, default=20, space="buy")
    kc_mult = DecimalParameter(1.0, 2.0, default=1.5, space="buy")
    mom_period = IntParameter(10, 20, default=12, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=self.bb_period.value, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]

        # Keltner Channels (EMA ± mult * ATR)
        dataframe["kc_mid"] = ta.EMA(dataframe, timeperiod=self.kc_period.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.kc_period.value)
        dataframe["kc_upper"] = dataframe["kc_mid"] + self.kc_mult.value * dataframe["atr"]
        dataframe["kc_lower"] = dataframe["kc_mid"] - self.kc_mult.value * dataframe["atr"]

        # Squeeze: BB inside KC = low volatility
        dataframe["squeeze_on"] = (
            (dataframe["bb_upper"] < dataframe["kc_upper"])
            & (dataframe["bb_lower"] > dataframe["kc_lower"])
        ).astype(int)
        dataframe["squeeze_prev"] = dataframe["squeeze_on"].shift(1)

        # Squeeze just released = was on, now off
        dataframe["squeeze_release"] = (
            (dataframe["squeeze_on"] == 0) & (dataframe["squeeze_prev"] == 1)
        ).astype(int)

        # Momentum: delta of midpoint of high/low and BB mid
        highest_high = dataframe["high"].rolling(self.mom_period.value).max()
        lowest_low = dataframe["low"].rolling(self.mom_period.value).min()
        mid_hl = (highest_high + lowest_low) / 2
        dataframe["momentum"] = dataframe["close"] - (mid_hl + dataframe["bb_mid"]) / 2
        dataframe["momentum_prev"] = dataframe["momentum"].shift(1)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["squeeze_release"] == 1)           # squeeze just fired
                & (dataframe["momentum"] > 0)                 # momentum is positive
                & (dataframe["momentum"] > dataframe["momentum_prev"])  # and rising
                & (dataframe["rsi"] > 50)                     # RSI confirms upside
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["momentum"] < 0)
                | (dataframe["rsi"] > 75)
            ),
            "exit_long",
        ] = 1
        return dataframe

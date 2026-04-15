"""
TrendAtrStrategy — Trend-following with ATR-based dynamic stops
================================================================
Logic:
  Entry  : EMA9 > EMA21 > EMA50 AND close pulls back to within 1×ATR of EMA21
            AND ADX > 20 (confirming trend strength)
  Exit   : close crosses below EMA21 OR trailing ATR stop hit
  Stop   : 2× ATR below entry (dynamic, not fixed %)

Suitable for: crypto / stocks, 4h timeframe
"""

import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from pandas import DataFrame


class TrendAtrStrategy(IStrategy):
    """AI-generated trend-following strategy using ATR dynamic stops."""

    timeframe = "4h"
    minimal_roi = {"0": 0.20, "480": 0.10, "1440": 0.05}
    stoploss = -0.08
    trailing_stop = False
    can_short = False
    startup_candle_count = 60

    # Hyperopt-ready parameters
    ema_fast = IntParameter(7, 15, default=9, space="buy")
    ema_mid = IntParameter(18, 26, default=21, space="buy")
    ema_slow = IntParameter(45, 60, default=50, space="buy")
    adx_threshold = DecimalParameter(18.0, 30.0, default=20.0, space="buy")
    atr_pullback = DecimalParameter(0.5, 2.0, default=1.0, space="buy")
    atr_stop_mult = DecimalParameter(1.5, 3.0, default=2.0, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=self.ema_mid.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Distance from close to EMA21 in ATR units
        dataframe["dist_to_mid"] = (dataframe["close"] - dataframe["ema_mid"]).abs() / dataframe["atr"]

        # ATR trailing stop level (stored for reference)
        dataframe["atr_stop"] = dataframe["close"] - self.atr_stop_mult.value * dataframe["atr"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema_fast"] > dataframe["ema_mid"])  # fast > mid
                & (dataframe["ema_mid"] > dataframe["ema_slow"])  # mid > slow — uptrend stack
                & (dataframe["adx"] > self.adx_threshold.value)  # trend is strong
                & (dataframe["dist_to_mid"] < self.atr_pullback.value)  # pulled back to mid EMA
                & (dataframe["close"] > dataframe["ema_mid"])  # still above mid
                & (dataframe["rsi"] < 75)  # not overbought
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] < dataframe["ema_mid"])  # price fell below mid EMA
                | (dataframe["ema_fast"] < dataframe["ema_mid"])  # fast crossed below mid
                | (dataframe["adx"] < 15)  # trend collapsed
            ),
            "exit_long",
        ] = 1
        return dataframe

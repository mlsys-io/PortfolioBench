"""
PriceActionCandleStrategy — Candlestick patterns + key level breakout
======================================================================
Logic:
  Detects bullish candlestick patterns (Hammer, Engulfing, Morning Star)
  at key support levels (near 20-bar low) with trend confirmation.

  Entry  : Any bullish candle pattern fires AND
            close is within 2% of 20-bar low (at support) AND
            EMA50 slope is positive (uptrend context)
  Exit   : Bearish engulfing OR price > upper Bollinger Band (target hit)
  Stop   : 4%
"""

import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IStrategy
from pandas import DataFrame


class PriceActionCandleStrategy(IStrategy):
    """Price action strategy using TA-Lib candlestick pattern recognition."""

    timeframe = "5m"
    minimal_roi = {"0": 0.12, "180": 0.06, "480": 0.03}
    stoploss = -0.04
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    can_short = False
    startup_candle_count = 60

    support_pct = DecimalParameter(0.01, 0.04, default=0.02, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Candlestick patterns
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["morning_star"] = ta.CDLMORNINGSTAR(dataframe)
        dataframe["piercing"] = ta.CDLPIERCING(dataframe)
        dataframe["dragonfly_doji"] = ta.CDLDRAGONFLYDOJI(dataframe)
        dataframe["bullish_pattern"] = (
            (dataframe["hammer"] > 0)
            | (dataframe["engulfing"] > 0)
            | (dataframe["morning_star"] > 0)
            | (dataframe["piercing"] > 0)
            | (dataframe["dragonfly_doji"] > 0)
        ).astype(int)

        # Bearish patterns for exit
        dataframe["bearish_engulfing"] = (dataframe["engulfing"] < 0).astype(int)
        dataframe["shooting_star"] = ta.CDLSHOOTINGSTAR(dataframe)

        # Key levels
        dataframe["low_20"] = dataframe["low"].rolling(20).min()
        dataframe["high_20"] = dataframe["high"].rolling(20).max()

        # Support proximity: close within X% of 20-bar low
        dataframe["near_support"] = (
            (dataframe["close"] - dataframe["low_20"]) / dataframe["low_20"].replace(0, 1)
            < self.support_pct.value
        ).astype(int)

        # Trend: EMA50 slope
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema50_prev"] = dataframe["ema50"].shift(3)
        dataframe["ema50_rising"] = (dataframe["ema50"] > dataframe["ema50_prev"]).astype(int)

        # Bollinger upper for target
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bullish_pattern"] == 1)
                & (dataframe["near_support"] == 1)
                & (dataframe["ema50_rising"] == 1)
                & (dataframe["rsi"] < 65)
                & (dataframe["volume"] > dataframe["volume_ma"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["bearish_engulfing"] == 1)
                | (dataframe["shooting_star"] > 0)
                | (dataframe["close"] > dataframe["bb_upper"])
                | (dataframe["rsi"] > 78)
            ),
            "exit_long",
        ] = 1
        return dataframe

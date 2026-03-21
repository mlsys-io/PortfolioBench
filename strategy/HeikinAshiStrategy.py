"""Strategy: Heikin Ashi Trend Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class HeikinAshiStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "240": 0.05}
    stoploss = -0.05
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        dataframe["ha_open"] = (dataframe["open"].shift(1) + dataframe["close"].shift(1)) / 2
        dataframe["ha_bullish"] = (dataframe["ha_close"] > dataframe["ha_open"]).astype(int)
        dataframe["ha_bullish_prev"] = dataframe["ha_bullish"].shift(1)
        dataframe["ha_bullish_prev2"] = dataframe["ha_bullish"].shift(2)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ha_bullish"] == 1)
            & (dataframe["ha_bullish_prev"] == 1)
            & (dataframe["ha_bullish_prev2"] == 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ha_bullish"] == 0) & (dataframe["ha_bullish_prev"] == 0),
            "exit_long",
        ] = 1
        return dataframe

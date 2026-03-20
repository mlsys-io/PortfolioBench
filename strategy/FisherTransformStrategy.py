"""Strategy: Fisher Transform Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np

class FisherTransformStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        period = 10
        highest = dataframe["high"].rolling(period).max()
        lowest = dataframe["low"].rolling(period).min()
        hlrange = (highest - lowest).replace(0, 0.001)
        value = 2 * ((dataframe["close"] - lowest) / hlrange) - 1
        value = value.clip(-0.999, 0.999)
        dataframe["fisher"] = 0.5 * np.log((1 + value) / (1 - value))
        dataframe["fisher_prev"] = dataframe["fisher"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["fisher"] > 0) & (dataframe["fisher_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["fisher"] > 2) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

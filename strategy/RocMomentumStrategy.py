"""Strategy 8: Rate of Change Momentum"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class RocMomentumStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=10)
        dataframe["roc_prev"] = dataframe["roc"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["roc"] > 0) & (dataframe["roc_prev"] <= 0)
            & (dataframe["ema50"] > dataframe["ema100"])
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 65)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["roc"] < 0) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

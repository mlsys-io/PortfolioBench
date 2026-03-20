"""Strategy: Higher Highs Higher Lows Structure"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class HigherHighStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["hh"] = (dataframe["high"] > dataframe["high"].shift(1)).astype(int)
        dataframe["hl"] = (dataframe["low"] > dataframe["low"].shift(1)).astype(int)
        dataframe["hh_prev"] = dataframe["hh"].shift(1)
        dataframe["hl_prev"] = dataframe["hl"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["hh"] == 1) & (dataframe["hl"] == 1)
            & (dataframe["hh_prev"] == 1) & (dataframe["hl_prev"] == 1)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["hh"] == 0) & (dataframe["hl"] == 0)
            | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy: ADX Strength Entry Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class AdxStrengthStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["adx_prev"] = dataframe["adx"].shift(1)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["adx"] > 25) & (dataframe["adx_prev"] <= 25)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["plus_di"] < dataframe["minus_di"])
            | (dataframe["adx"] < 15)
            | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy: On Balance Volume Trend"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ObvTrendStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_ema"] = ta.EMA(dataframe["obv"], timeperiod=20)
        dataframe["obv_ema_prev"] = dataframe["obv_ema"].shift(1)
        dataframe["obv_above"] = (dataframe["obv"] > dataframe["obv_ema"]).astype(int)
        dataframe["obv_above_prev"] = dataframe["obv_above"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["obv_above"] == 1) & (dataframe["obv_above_prev"] == 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["obv_above"] == 0) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy: TEMA Cross Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class TemaStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 35

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["tema20"] = ta.TEMA(dataframe, timeperiod=20)
        dataframe["tema50"] = ta.TEMA(dataframe, timeperiod=50)
        dataframe["tema20_prev"] = dataframe["tema20"].shift(1)
        dataframe["tema50_prev"] = dataframe["tema50"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["tema20"] > dataframe["tema50"])
            & (dataframe["tema20_prev"] <= dataframe["tema50_prev"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["tema20"] < dataframe["tema50"]) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

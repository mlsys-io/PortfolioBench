"""Strategy: Dema Cross Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class DemaCrossStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["dema20"] = ta.DEMA(dataframe, timeperiod=20)
        dataframe["dema50"] = ta.DEMA(dataframe, timeperiod=50)
        dataframe["dema20_prev"] = dataframe["dema20"].shift(1)
        dataframe["dema50_prev"] = dataframe["dema50"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["dema20"] > dataframe["dema50"])
            & (dataframe["dema20_prev"] <= dataframe["dema50_prev"])
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["dema20"] < dataframe["dema50"]) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

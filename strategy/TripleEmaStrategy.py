"""Strategy 11: Triple EMA Trend"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class TripleEmaStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "240": 0.05}
    stoploss = -0.05
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["ema8_prev"] = dataframe["ema8"].shift(1)
        dataframe["ema21_prev"] = dataframe["ema21"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ema8"] > dataframe["ema21"])
            & (dataframe["ema21"] > dataframe["ema55"])
            & (dataframe["ema8_prev"] <= dataframe["ema21_prev"])
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ema8"] < dataframe["ema21"]) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy 7: Williams %R Reversal"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class WilliamsRStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["willr"] = ta.WILLR(dataframe, timeperiod=14)
        dataframe["willr_prev"] = dataframe["willr"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["willr"] > -80) & (dataframe["willr_prev"] <= -80)
            & (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["rsi"] < 65)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["willr"] > -20) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

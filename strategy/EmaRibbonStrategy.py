"""Strategy: EMA Ribbon Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class EmaRibbonStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 40

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema13"] = ta.EMA(dataframe, timeperiod=13)
        dataframe["ema21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema34"] = ta.EMA(dataframe, timeperiod=34)
        dataframe["ema55"] = ta.EMA(dataframe, timeperiod=55)
        dataframe["ribbon_bull"] = (
            (dataframe["ema8"] > dataframe["ema13"])
            & (dataframe["ema13"] > dataframe["ema21"])
            & (dataframe["ema21"] > dataframe["ema34"])
            & (dataframe["ema34"] > dataframe["ema55"])
        ).astype(int)
        dataframe["ribbon_bull_prev"] = dataframe["ribbon_bull"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ribbon_bull"] == 1) & (dataframe["ribbon_bull_prev"] == 0)
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ribbon_bull"] == 0) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

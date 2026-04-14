"""Strategy: Bullish Engulfing Pattern Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class BullishEngulfingStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["engulfing"] = ta.CDLENGULFING(dataframe)
        dataframe["hammer"] = ta.CDLHAMMER(dataframe)
        dataframe["morning_doji"] = ta.CDLMORNINGDOJISTAR(dataframe)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            ((dataframe["engulfing"] > 0) | (dataframe["hammer"] > 0) | (dataframe["morning_doji"] > 0))
            & (dataframe["ema50"] > dataframe["ema100"])
            & (dataframe["rsi"] < 60)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["engulfing"] < 0) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

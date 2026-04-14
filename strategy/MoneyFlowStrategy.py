"""Strategy 10: Money Flow Index"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class MoneyFlowStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["mfi_prev"] = dataframe["mfi"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["mfi"] > 20) & (dataframe["mfi_prev"] <= 20)
            & (dataframe["ema50"] > dataframe["ema100"])
            & (dataframe["rsi"] < 65)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["mfi"] > 80) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

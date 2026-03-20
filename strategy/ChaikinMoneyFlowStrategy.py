"""Strategy: Chaikin Money Flow Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class ChaikinMoneyFlowStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["cmf"] = ta.ADOSC(dataframe, fastperiod=3, slowperiod=10)
        dataframe["cmf_prev"] = dataframe["cmf"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["cmf"] > 0) & (dataframe["cmf_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 65)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["cmf"] < 0) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

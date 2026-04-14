"""Strategy: Trix Signal Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class TrixSignalStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 40

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["trix"] = ta.TRIX(dataframe, timeperiod=15)
        dataframe["trix_prev"] = dataframe["trix"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["trix"] > 0) & (dataframe["trix_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["trix"] < 0) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

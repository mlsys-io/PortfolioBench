"""Strategy 4: MACD Zero Cross"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class MacdZeroCrossStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 35

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_prev"] = dataframe["macd"].shift(1)
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["macd"] > 0) & (dataframe["macd_prev"] <= 0)
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 65)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["macd"] < 0) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

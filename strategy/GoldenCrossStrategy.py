"""Strategy 2: Golden Cross Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class GoldenCrossStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.12, "240": 0.06}
    stoploss = -0.05
    startup_candle_count = 55

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema50_prev"] = dataframe["ema50"].shift(1)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["ema200_prev"] = dataframe["ema200"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ema50"] > dataframe["ema200"])
            & (dataframe["ema50_prev"] <= dataframe["ema200_prev"])
            & (dataframe["rsi"] < 70)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ema50"] < dataframe["ema200"]) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

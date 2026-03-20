"""Strategy: Momentum Score Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class MomentumScoreStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=10)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        dataframe["score"] = (
            (dataframe["rsi"] > 50).astype(int)
            + (dataframe["adx"] > 20).astype(int)
            + (dataframe["ema20"] > dataframe["ema50"]).astype(int)
            + (dataframe["roc"] > 0).astype(int)
            + (dataframe["volume"] > dataframe["volume_ma"]).astype(int)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["score"] >= 4)
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["score"] <= 2) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

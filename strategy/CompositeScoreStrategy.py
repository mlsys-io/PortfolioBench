"""Strategy 13: Composite Score Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class CompositeScoreStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd_hist"] = macd["macdhist"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()

        # Simple score: 1 point each
        dataframe["score"] = (
            (dataframe["rsi"] > 45).astype(int)
            + (dataframe["rsi"] < 65).astype(int)
            + (dataframe["ema20"] > dataframe["ema50"]).astype(int)
            + (dataframe["macd_hist"] > 0).astype(int)
            + (dataframe["adx"] > 20).astype(int)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["score"] >= 4)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["score"] <= 2) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

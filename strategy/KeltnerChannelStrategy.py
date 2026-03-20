"""Strategy: Keltner Channel Breakout"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class KeltnerChannelStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["kc_upper"] = dataframe["ema20"] + 2 * dataframe["atr"]
        dataframe["kc_lower"] = dataframe["ema20"] - 2 * dataframe["atr"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["kc_upper"])
            & (dataframe["adx"] > 25)
            & (dataframe["rsi"] > 50) & (dataframe["rsi"] < 70)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["ema20"]) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy: Price Channel Trend Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class PriceChannelStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "240": 0.05}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["upper_channel"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["lower_channel"] = dataframe["low"].rolling(20).min().shift(1)
        dataframe["mid_channel"] = (dataframe["upper_channel"] + dataframe["lower_channel"]) / 2
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["upper_channel"])
            & (dataframe["adx"] > 22)
            & (dataframe["rsi"] > 50) & (dataframe["rsi"] < 70)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["mid_channel"]) | (dataframe["rsi"] > 76),
            "exit_long",
        ] = 1
        return dataframe

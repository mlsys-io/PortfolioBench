"""Strategy 12: Donchian Channel Breakout"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class DonchianBreakoutStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.12, "240": 0.06}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["dc_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["dc_low"] = dataframe["low"].rolling(20).min().shift(1)
        dataframe["dc_mid"] = (dataframe["dc_high"] + dataframe["dc_low"]) / 2
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["dc_high"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"] > 50) & (dataframe["rsi"] < 72)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["dc_mid"]) | (dataframe["rsi"] > 78),
            "exit_long",
        ] = 1
        return dataframe

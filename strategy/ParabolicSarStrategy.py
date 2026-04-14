"""Strategy 9: Parabolic SAR Trend Follow"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class ParabolicSarStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "240": 0.05}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sar"] = ta.SAR(dataframe, acceleration=0.02, maximum=0.2)
        dataframe["sar_prev"] = dataframe["sar"].shift(1)
        dataframe["close_prev"] = dataframe["close"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["sar"])
            & (dataframe["close_prev"] <= dataframe["sar_prev"])
            & (dataframe["adx"] > 20)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["sar"]) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

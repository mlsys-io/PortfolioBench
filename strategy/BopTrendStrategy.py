"""Strategy: Balance of Power Trend Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class BopTrendStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["bop"] = ta.BOP(dataframe)
        dataframe["bop_ma"] = dataframe["bop"].rolling(10).mean()
        dataframe["bop_ma_prev"] = dataframe["bop_ma"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["bop_ma"] > 0) & (dataframe["bop_ma_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["adx"] > 18)
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["bop_ma"] < -0.1) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

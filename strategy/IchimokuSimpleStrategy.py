"""Strategy: Simple Ichimoku Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class IchimokuSimpleStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.12, "240": 0.06}
    stoploss = -0.05
    startup_candle_count = 60

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        high9 = dataframe["high"].rolling(9).max()
        low9 = dataframe["low"].rolling(9).min()
        dataframe["tenkan"] = (high9 + low9) / 2
        high26 = dataframe["high"].rolling(26).max()
        low26 = dataframe["low"].rolling(26).min()
        dataframe["kijun"] = (high26 + low26) / 2
        dataframe["senkou_a"] = ((dataframe["tenkan"] + dataframe["kijun"]) / 2).shift(26)
        high52 = dataframe["high"].rolling(52).max()
        low52 = dataframe["low"].rolling(52).min()
        dataframe["senkou_b"] = ((high52 + low52) / 2).shift(26)
        dataframe["tenkan_prev"] = dataframe["tenkan"].shift(1)
        dataframe["kijun_prev"] = dataframe["kijun"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["tenkan"] > dataframe["kijun"])
            & (dataframe["tenkan_prev"] <= dataframe["kijun_prev"])
            & (dataframe["close"] > dataframe["senkou_a"])
            & (dataframe["close"] > dataframe["senkou_b"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["tenkan"] < dataframe["kijun"]) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

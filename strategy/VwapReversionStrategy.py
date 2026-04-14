"""Strategy: VWAP Reversion Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class VwapReversionStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tp = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe["vwap"] = (tp * dataframe["volume"]).rolling(20).sum() / dataframe["volume"].rolling(20).sum()
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["below_vwap"] = (dataframe["close"] < dataframe["vwap"]).astype(int)
        dataframe["below_vwap_prev"] = dataframe["below_vwap"].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["below_vwap"] == 0) & (dataframe["below_vwap_prev"] == 1)
            & (dataframe["close"] > dataframe["ema100"])
            & (dataframe["rsi"] < 65)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["vwap"] * 1.02) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

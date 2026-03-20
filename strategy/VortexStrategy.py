"""Strategy: Vortex Indicator Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class VortexStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        period = 14
        tr = dataframe["high"] - dataframe["low"]
        vm_plus = (dataframe["high"] - dataframe["low"].shift(1)).abs()
        vm_minus = (dataframe["low"] - dataframe["high"].shift(1)).abs()
        dataframe["vi_plus"] = vm_plus.rolling(period).sum() / tr.rolling(period).sum().replace(0, 1)
        dataframe["vi_minus"] = vm_minus.rolling(period).sum() / tr.rolling(period).sum().replace(0, 1)
        dataframe["vi_plus_prev"] = dataframe["vi_plus"].shift(1)
        dataframe["vi_minus_prev"] = dataframe["vi_minus"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["vi_plus"] > dataframe["vi_minus"])
            & (dataframe["vi_plus_prev"] <= dataframe["vi_minus_prev"])
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["vi_plus"] < dataframe["vi_minus"]) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

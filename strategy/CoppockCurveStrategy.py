"""Strategy: Coppock Curve Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class CoppockCurveStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "240": 0.05}
    stoploss = -0.05
    startup_candle_count = 40

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        roc14 = ta.ROC(dataframe, timeperiod=14)
        roc11 = ta.ROC(dataframe, timeperiod=11)
        dataframe["coppock"] = ta.WMA(roc14 + roc11, timeperiod=10)
        dataframe["coppock_prev"] = dataframe["coppock"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["coppock"] > 0) & (dataframe["coppock_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["coppock"] < 0) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

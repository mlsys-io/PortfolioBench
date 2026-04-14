"""Strategy: Linear Regression Slope Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class LinearRegressionStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["linreg"] = ta.LINEARREG(dataframe, timeperiod=14)
        dataframe["linreg_slope"] = ta.LINEARREG_SLOPE(dataframe, timeperiod=14)
        dataframe["linreg_slope_prev"] = dataframe["linreg_slope"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["linreg_slope"] > 0) & (dataframe["linreg_slope_prev"] <= 0)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] > 45) & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["linreg_slope"] < 0) | (dataframe["rsi"] > 73),
            "exit_long",
        ] = 1
        return dataframe

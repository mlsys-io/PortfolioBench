"""Strategy: Stochastic RSI Strategy"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class StochasticRsiStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        rsi_min = dataframe["rsi"].rolling(14).min()
        rsi_max = dataframe["rsi"].rolling(14).max()
        rsi_range = (rsi_max - rsi_min).replace(0, 1)
        dataframe["stoch_rsi"] = (dataframe["rsi"] - rsi_min) / rsi_range * 100
        dataframe["stoch_rsi_prev"] = dataframe["stoch_rsi"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["stoch_rsi"] > 20) & (dataframe["stoch_rsi_prev"] <= 20)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["stoch_rsi"] > 80) | (dataframe["rsi"] > 72),
            "exit_long",
        ] = 1
        return dataframe

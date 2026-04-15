"""Strategy 5: Stochastic Oversold Bounce"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class StochasticOversoldStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 20

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]
        dataframe["stoch_k_prev"] = dataframe["stoch_k"].shift(1)
        dataframe["stoch_d_prev"] = dataframe["stoch_d"].shift(1)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["stoch_k"] > dataframe["stoch_d"])
            & (dataframe["stoch_k_prev"] <= dataframe["stoch_d_prev"])
            & (dataframe["stoch_k"] < 30)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["stoch_k"] > 80) | (dataframe["rsi"] > 70),
            "exit_long",
        ] = 1
        return dataframe

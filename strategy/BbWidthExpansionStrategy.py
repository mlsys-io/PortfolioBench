"""Strategy: Bollinger Band Width Expansion Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class BbWidthExpansionStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_width"] = (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_mid"]
        dataframe["bb_width_prev"] = dataframe["bb_width"].shift(1)
        dataframe["bb_width_ma"] = dataframe["bb_width"].rolling(20).mean()
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["bb_width"] > dataframe["bb_width_ma"])
            & (dataframe["bb_width_prev"] <= dataframe["bb_width_ma"])
            & (dataframe["close"] > dataframe["bb_mid"])
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["rsi"] > 50) & (dataframe["rsi"] < 70)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_mid"]) | (dataframe["rsi"] > 74),
            "exit_long",
        ] = 1
        return dataframe

"""Strategy 6: CCI Mean Reversion"""
import talib.abstract as ta
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class CciMeanReversionStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["cci_prev"] = dataframe["cci"].shift(1)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["cci"] > -100) & (dataframe["cci_prev"] <= -100)
            & (dataframe["close"] > dataframe["ema100"])
            & (dataframe["rsi"] < 60)
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["cci"] > 100) | (dataframe["rsi"] > 70),
            "exit_long",
        ] = 1
        return dataframe

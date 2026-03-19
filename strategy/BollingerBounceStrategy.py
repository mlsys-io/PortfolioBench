"""Strategy 3: Bollinger Band Bounce"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class BollingerBounceStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.08, "120": 0.04}
    stoploss = -0.04
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_mid"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema100"] = ta.EMA(dataframe, timeperiod=100)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["rsi"] < 40)
            & (dataframe["close"] > dataframe["ema100"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_mid"]) | (dataframe["rsi"] > 68),
            "exit_long",
        ] = 1
        return dataframe

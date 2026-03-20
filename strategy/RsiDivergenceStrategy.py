"""Strategy: RSI Divergence Proxy Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class RsiDivergenceStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.10, "180": 0.05}
    stoploss = -0.05
    startup_candle_count = 25

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_min"] = dataframe["rsi"].rolling(10).min()
        dataframe["price_min"] = dataframe["close"].rolling(10).min()
        dataframe["rsi_rising"] = (dataframe["rsi"] > dataframe["rsi"].shift(3)).astype(int)
        dataframe["price_falling"] = (dataframe["close"] < dataframe["close"].shift(3)).astype(int)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] < 45)
            & (dataframe["rsi_rising"] == 1)
            & (dataframe["close"] > dataframe["ema50"])
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] > 70),
            "exit_long",
        ] = 1
        return dataframe

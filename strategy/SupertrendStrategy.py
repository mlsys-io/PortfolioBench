"""Strategy: Supertrend Strategy"""
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class SupertrendStrategy(IStrategy):
    timeframe = "5m"
    minimal_roi = {"0": 0.12, "240": 0.06}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    startup_candle_count = 25

    def _supertrend(self, dataframe, period=10, multiplier=3.0):
        atr = ta.ATR(dataframe, timeperiod=period)
        hl2 = (dataframe["high"] + dataframe["low"]) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        supertrend = lower.copy()
        direction = [1] * len(dataframe)
        for i in range(1, len(dataframe)):
            if dataframe["close"].iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = max(lower.iloc[i], supertrend.iloc[i - 1])
                direction[i] = 1
            else:
                supertrend.iloc[i] = min(upper.iloc[i], supertrend.iloc[i - 1])
                direction[i] = -1
        return supertrend, direction

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        st, direction = self._supertrend(dataframe)
        dataframe["supertrend"] = st
        dataframe["st_dir"] = direction
        dataframe["st_dir_prev"] = dataframe["st_dir"].shift(1)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_ma"] = dataframe["volume"].rolling(20).mean()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["st_dir"] == 1) & (dataframe["st_dir_prev"] == -1)
            & (dataframe["rsi"] < 68)
            & (dataframe["volume"] > dataframe["volume_ma"])
            & (dataframe["volume"] > 0),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["st_dir"] == -1) | (dataframe["rsi"] > 75),
            "exit_long",
        ] = 1
        return dataframe

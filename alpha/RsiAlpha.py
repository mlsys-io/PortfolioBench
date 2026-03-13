"""RSI (Relative Strength Index) alpha factor.

Computes the classic Wilder RSI along with overbought/oversold zone flags
and a smoothed RSI signal line useful for crossover strategies.
"""

import talib.abstract as ta

from alpha.interface import IAlpha
from freqtrade.strategy.parameters import IntParameter


class RsiAlpha(IAlpha):
    def __init__(self, dataframe, metadata=None):
        self.rsi_period = IntParameter(7, 21, default=14, space="buy")
        self.rsi_signal_period = IntParameter(3, 10, default=9, space="buy")
        super().__init__(dataframe, metadata)

    def process(self):
        self.dataframe["rsi"] = ta.RSI(self.dataframe, timeperiod=self.rsi_period.value)
        self.dataframe["rsi_signal"] = (
            self.dataframe["rsi"].rolling(self.rsi_signal_period.value).mean()
        )
        self.dataframe["rsi_overbought"] = (self.dataframe["rsi"] > 70).astype(int)
        self.dataframe["rsi_oversold"] = (self.dataframe["rsi"] < 30).astype(int)
        self.dataframe["mean-volume"] = self.dataframe["volume"].rolling(20).mean()
        return self.dataframe

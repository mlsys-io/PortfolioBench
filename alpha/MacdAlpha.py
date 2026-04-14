"""MACD (Moving Average Convergence Divergence) alpha factor.

Computes the MACD line, signal line, and histogram. Also includes a
histogram-direction flag useful for momentum strategies.
"""

import talib.abstract as ta
from freqtrade.strategy.parameters import IntParameter

from alpha.interface import IAlpha


class MacdAlpha(IAlpha):
    def __init__(self, dataframe, metadata=None):
        self.fast_period = IntParameter(8, 15, default=12, space="buy")
        self.slow_period = IntParameter(20, 30, default=26, space="buy")
        self.signal_period = IntParameter(5, 12, default=9, space="buy")
        super().__init__(dataframe, metadata)

    def process(self):
        macd = ta.MACD(
            self.dataframe,
            fastperiod=self.fast_period.value,
            slowperiod=self.slow_period.value,
            signalperiod=self.signal_period.value,
        )
        self.dataframe["macd"] = macd["macd"]
        self.dataframe["macd_signal"] = macd["macdsignal"]
        self.dataframe["macd_hist"] = macd["macdhist"]
        self.dataframe["macd_hist_rising"] = (
            self.dataframe["macd_hist"].diff().gt(0).astype(int)
        )
        self.dataframe["mean-volume"] = self.dataframe["volume"].rolling(20).mean()
        return self.dataframe

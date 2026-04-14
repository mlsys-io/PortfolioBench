import talib.abstract as ta
from freqtrade.strategy.parameters import IntParameter

from alpha.interface import IAlpha


class EmaAlpha(IAlpha):
    def __init__(self, dataframe, metadata):
        self.ema_fast_length = IntParameter(5, 15, default=12, space="buy")
        self.ema_slow_length = IntParameter(20, 30, default=26, space="buy")
        self.ema_exit_length = IntParameter(5, 10, default=6, space="sell")
        super().__init__(dataframe, metadata)
        
    def process(self):
        self.dataframe["ema_fast"] = ta.EMA(self.dataframe, timeperiod=self.ema_fast_length.value)
        self.dataframe["ema_slow"] = ta.EMA(self.dataframe, timeperiod=self.ema_slow_length.value)
        self.dataframe["ema_exit"] = ta.EMA(self.dataframe, timeperiod=self.ema_exit_length.value)
        self.dataframe["mean-volume"] = self.dataframe["volume"].rolling(20).mean()
        return self.dataframe
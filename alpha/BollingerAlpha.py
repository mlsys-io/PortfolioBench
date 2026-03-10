"""Bollinger Bands alpha factor.

Computes upper, middle (SMA), and lower Bollinger Bands along with the
bandwidth and percent-B (%B) indicators. %B measures where price sits
relative to the bands (0 = lower band, 1 = upper band).
"""

import talib.abstract as ta

from alpha.interface import IAlpha
from freqtrade.strategy.parameters import IntParameter


class BollingerAlpha(IAlpha):
    def __init__(self, dataframe, metadata=None):
        self.bb_period = IntParameter(10, 30, default=20, space="buy")
        self.bb_nbdev = 2.0
        super().__init__(dataframe, metadata)

    def process(self):
        bb = ta.BBANDS(
            self.dataframe,
            timeperiod=self.bb_period.value,
            nbdevup=self.bb_nbdev,
            nbdevdn=self.bb_nbdev,
        )
        self.dataframe["bb_upper"] = bb["upperband"]
        self.dataframe["bb_middle"] = bb["middleband"]
        self.dataframe["bb_lower"] = bb["lowerband"]

        band_width = self.dataframe["bb_upper"] - self.dataframe["bb_lower"]
        self.dataframe["bb_width"] = band_width / self.dataframe["bb_middle"]

        self.dataframe["bb_pctb"] = (
            (self.dataframe["close"] - self.dataframe["bb_lower"]) /
            band_width.replace(0, float("nan"))
        )

        self.dataframe["mean-volume"] = self.dataframe["volume"].rolling(20).mean()
        return self.dataframe

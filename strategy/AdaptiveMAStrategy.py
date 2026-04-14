import numpy as np

"""
AdaptiveMAStrategy — Kaufman Adaptive Moving Average (KAMA) crossover
=====================================================================
Logic:
  KAMA adapts its speed based on market efficiency — fast in trending
  markets, slow in choppy markets. This reduces false signals vs
  standard EMA crossovers.

  Entry  : KAMA fast crosses above KAMA slow AND
            Efficiency Ratio > 0.3 (trending market) AND
            RSI 45-65 (healthy momentum, not extreme)
  Exit   : KAMA fast crosses below KAMA slow OR RSI > 72
  Stop   : 5%
"""

import talib.abstract as ta
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy
from pandas import DataFrame


class AdaptiveMAStrategy(IStrategy):
    """Adaptive moving average crossover using KAMA."""

    timeframe = "5m"
    minimal_roi = {"0": 0.15, "300": 0.08, "720": 0.04}
    stoploss = -0.05
    trailing_stop = False
    can_short = False
    startup_candle_count = 50

    kama_fast_period = IntParameter(5, 15, default=8, space="buy")
    kama_slow_period = IntParameter(20, 40, default=30, space="buy")
    er_period = IntParameter(8, 20, default=10, space="buy")
    er_threshold = DecimalParameter(0.2, 0.5, default=0.3, space="buy")
    rsi_min = IntParameter(40, 52, default=45, space="buy")
    rsi_max = IntParameter(60, 72, default=65, space="sell")

    def _kama(self, close, period=10, fast=2, slow=30):
        """Compute Kaufman Adaptive Moving Average."""
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        kama = close.copy()
        for i in range(period, len(close)):
            direction = abs(close.iloc[i] - close.iloc[i - period])
            volatility = sum(
                abs(close.iloc[j] - close.iloc[j - 1]) for j in range(i - period + 1, i + 1)
            )
            er = direction / volatility if volatility != 0 else 0
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            kama.iloc[i] = kama.iloc[i - 1] + sc * (close.iloc[i] - kama.iloc[i - 1])
        return kama

    def _efficiency_ratio(self, close, period):
        """Market Efficiency Ratio: 1 = perfectly trending, 0 = random."""
        direction = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(period).sum()
        return direction / volatility.replace(0, np.nan)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # KAMA fast and slow
        dataframe["kama_fast"] = self._kama(
            dataframe["close"], period=self.kama_fast_period.value, fast=2, slow=20
        )
        dataframe["kama_slow"] = self._kama(
            dataframe["close"], period=self.kama_slow_period.value, fast=2, slow=30
        )

        dataframe["kama_fast_prev"] = dataframe["kama_fast"].shift(1)
        dataframe["kama_slow_prev"] = dataframe["kama_slow"].shift(1)

        # Efficiency Ratio
        dataframe["er"] = self._efficiency_ratio(dataframe["close"], self.er_period.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # Volume filter
        dataframe["vol_ma"] = dataframe["volume"].rolling(20).mean()

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # KAMA fast crosses above KAMA slow
                (dataframe["kama_fast"] > dataframe["kama_slow"])
                & (dataframe["kama_fast_prev"] <= dataframe["kama_slow_prev"])
                # Market is trending (not choppy)
                & (dataframe["er"] > self.er_threshold.value)
                # RSI in healthy zone
                & (dataframe["rsi"] > self.rsi_min.value)
                & (dataframe["rsi"] < self.rsi_max.value)
                # Volume confirms
                & (dataframe["volume"] > dataframe["vol_ma"])
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # KAMA fast crosses below KAMA slow
                (
                    (dataframe["kama_fast"] < dataframe["kama_slow"])
                    & (dataframe["kama_fast_prev"] >= dataframe["kama_slow_prev"])
                )
                | (dataframe["rsi"] > self.rsi_max.value + 7)
                | (dataframe["adx"] < 15)  # trend collapsed
            ),
            "exit_long",
        ] = 1
        return dataframe

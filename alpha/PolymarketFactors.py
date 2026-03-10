"""Alpha factors for Polymarket event contract analysis.

Polymarket contracts are probability-priced [0, 1] binary outcomes.
Traditional momentum/mean-reversion indicators need adaptation:
- Prices are bounded, so unbounded indicators can mislead
- Probability space means RSI-like indicators are natural fits
- Volume spikes often precede resolution or major news
- Complementary contracts (YES+NO=1) create natural arbitrage signals
"""

import numpy as np
from pandas import DataFrame

from alpha.interface import IAlpha


class PolymarketAlpha(IAlpha):
    """Alpha factors tailored for binary outcome prediction market contracts.

    Indicators produced:
    - prob_momentum: Rate of probability change (smoothed)
    - prob_rsi: RSI adapted for probability space
    - volume_surge: Volume relative to rolling average (news detection)
    - prob_zscore: Z-score of current price vs rolling window
    - mean_reversion_signal: Distance from rolling mean (mean-reversion)
    - resolution_proximity: How close price is to 0 or 1 (conviction signal)
    - prob_ema_fast: Fast EMA of probability
    - prob_ema_slow: Slow EMA of probability
    """

    def __init__(self, dataframe: DataFrame, metadata: dict = None):
        self.fast_period = 12
        self.slow_period = 26
        self.rsi_period = 14
        self.zscore_period = 20
        self.volume_period = 20
        super().__init__(dataframe, metadata)

    def process(self) -> DataFrame:
        df = self.dataframe

        close = df["close"]

        # --- Probability EMAs ---
        df["prob_ema_fast"] = close.ewm(span=self.fast_period, adjust=False).mean()
        df["prob_ema_slow"] = close.ewm(span=self.slow_period, adjust=False).mean()

        # --- Probability Momentum ---
        # Rate of change in probability, smoothed
        df["prob_momentum"] = close.diff(5).rolling(3).mean()

        # --- RSI in probability space ---
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["prob_rsi"] = 100 - (100 / (1 + rs))
        df["prob_rsi"] = df["prob_rsi"].fillna(50)

        # --- Volume Surge ---
        # Ratio of current volume to rolling average; >2 suggests news
        mean_vol = df["volume"].rolling(self.volume_period).mean()
        df["volume_surge"] = df["volume"] / mean_vol.replace(0, 1)

        # --- Z-score of probability ---
        rolling_mean = close.rolling(self.zscore_period).mean()
        rolling_std = close.rolling(self.zscore_period).std()
        df["prob_zscore"] = (close - rolling_mean) / rolling_std.replace(0, np.nan)
        df["prob_zscore"] = df["prob_zscore"].fillna(0)

        # --- Mean reversion signal ---
        # Distance from rolling mean, normalized. Positive = above mean
        df["mean_reversion_signal"] = close - rolling_mean

        # --- Resolution proximity ---
        # How close to 0 or 1: min(price, 1-price). Low = high conviction
        df["resolution_proximity"] = np.minimum(close, 1 - close)

        # --- Volume-weighted mean volume for volume filter ---
        df["mean_volume"] = mean_vol

        return df

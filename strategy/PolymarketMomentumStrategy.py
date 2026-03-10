"""Polymarket Momentum Strategy — trend-following on probability shifts.

Trades event contracts by following probability momentum:
- Enter YES when probability is rising and momentum confirms uptrend
- Exit when momentum reverses or contract approaches resolution

Designed for contracts where information is gradually incorporated
(e.g., election polls, economic forecasts).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from freqtrade.strategy import IStrategy, Trade

from alpha.PolymarketFactors import PolymarketAlpha


class PolymarketMomentumStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False
    minimal_roi = {}
    stoploss = -0.50  # Wide stop: contracts can be volatile
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.10

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = PolymarketAlpha(dataframe, metadata).process()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Momentum is positive (probability rising)
                (dataframe["prob_momentum"] > 0.002)
                # Fast EMA above slow EMA (uptrend)
                & (dataframe["prob_ema_fast"] > dataframe["prob_ema_slow"])
                # RSI not overbought (room to run)
                & (dataframe["prob_rsi"] < 75)
                # Price not too close to resolution (avoid buying at 0.95+)
                & (dataframe["close"] < 0.85)
                # Price not too cheap (avoid noise at very low prob)
                & (dataframe["close"] > 0.10)
                # Some volume present
                & (dataframe["mean_volume"] > 0)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Momentum turned negative
                (dataframe["prob_momentum"] < -0.002)
                # OR RSI overbought
                | (dataframe["prob_rsi"] > 80)
                # OR price very close to 1 (take profit near resolution)
                | (dataframe["close"] > 0.95)
            ),
            "exit_long",
        ] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        # Reject entries if the contract price is outside tradeable range
        if rate < 0.05 or rate > 0.95:
            return False
        return True

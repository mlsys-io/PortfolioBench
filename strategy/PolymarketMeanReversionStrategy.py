"""Polymarket Mean-Reversion Strategy — fade overreactions in prediction markets.

Trades event contracts by betting against short-term overreactions:
- Buy when probability drops significantly below its rolling mean (oversold)
- Sell when probability reverts to mean or overshoots above it

Designed for contracts where sharp moves are driven by noise/overreaction
rather than fundamental shifts (e.g., speculative events, sentiment spikes).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from freqtrade.strategy import IStrategy, Trade

from alpha.PolymarketFactors import PolymarketAlpha


class PolymarketMeanReversionStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short: bool = False
    minimal_roi = {"0": 0.15}  # Take profit at 15% gain
    stoploss = -0.30
    trailing_stop = False

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
                # Z-score strongly negative (price well below mean)
                (dataframe["prob_zscore"] < -1.5)
                # Mean reversion signal confirms (below rolling mean)
                & (dataframe["mean_reversion_signal"] < -0.03)
                # Volume surge suggests reactionary move, not fundamental
                & (dataframe["volume_surge"] > 1.5)
                # Contract still has room to move (not near resolution)
                & (dataframe["resolution_proximity"] > 0.10)
                # Price in tradeable range
                & (dataframe["close"] > 0.10)
                & (dataframe["close"] < 0.90)
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Price reverted above mean
                (dataframe["prob_zscore"] > 0.5)
                # OR momentum shifted positive (reversion complete)
                | (
                    (dataframe["mean_reversion_signal"] > 0.02)
                    & (dataframe["prob_momentum"] > 0)
                )
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
        if rate < 0.05 or rate > 0.95:
            return False
        return True

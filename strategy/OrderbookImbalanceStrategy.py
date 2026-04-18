"""Orderbook Imbalance Strategy — baseline proof-of-concept for Polymarket CLOB features.

Signal: when the EMA-smoothed top-3 orderbook imbalance is positive and rising,
enter long. Exit when it turns negative.

Works with any pair that has a corresponding feat_orderbook parquet file.
Token-id resolution is automatic via freqtrade_pair_mapping.csv in data.zip.

Backtest command (Panthers YES example):
----------------------------------------
    portbench backtesting \\
        --strategy OrderbookImbalanceStrategy \\
        --strategy-path ./strategy \\
        --config user_data/config_polymarket.json \\
        --datadir user_data/data/polymarket/data \\
        --timeframe 4h \\
        --timerange 20251015-20260112 \\
        --pairs WillTheCarolinaPanthersWinSuperBowYES20250430/USDC
"""

import pandas as pd
from freqtrade.strategy import IStrategy

from alpha.OrderbookAlpha import OrderbookAlpha


class OrderbookImbalanceStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = False
    minimal_roi = {"0": 0.40}
    stoploss = -0.20
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    startup_candle_count = 10

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # OrderbookAlpha resolves the token_id automatically from metadata["pair"]
        dataframe = OrderbookAlpha(dataframe, metadata).process()
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (dataframe["ob_imbalance_ema"] > 0.10)
            & (dataframe["ob_imbalance_ema"] > dataframe["ob_imbalance_ema"].shift(1))
            & (dataframe["close"] < 0.95),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            dataframe["ob_imbalance_ema"] < -0.05,
            "exit_long",
        ] = 1
        return dataframe

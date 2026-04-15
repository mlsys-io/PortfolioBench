import pandas as pd
from freqtrade.strategy import IStrategy
from pandas import DataFrame


class BestSingleAssetPortfolio(IStrategy):
    """
    Winner-takes-all momentum rotation strategy
    """
    INTERFACE_VERSION = 3
    timeframe = "1d"
    stoploss = -1.0

    # Portfolio-style behavior
    max_open_trades = 1
    REBALANCE_DATE = 1
    LOOKBACK = 90
    process_only_new_candles = True
    startup_candle_count = 90 # Ensure enough data for lookback

    minimal_roi = {}

    def informative_pairs(self):
        """
        Load all whitelist pairs as informative pairs
        so every pair can see every other pair.
        """
        return [(pair, self.timeframe) for pair in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        For each candle, determine which pair has the best return in LOOKBACK days.
        """
        
        dataframe["momentum"] = dataframe["close"] / dataframe["close"].shift(self.LOOKBACK) - 1
        dataframe["rebalance"] = pd.to_datetime(dataframe["date"]).dt.day == self.REBALANCE_DATE
        
        momentum_map = {}

        for pair in self.dp.current_whitelist():
            inf_df = self.dp.get_pair_dataframe(pair, self.timeframe)
            momentum_map[pair] = (
                inf_df["close"] / inf_df["close"].shift(self.LOOKBACK) - 1
            )

        momentum_df = pd.DataFrame(momentum_map)

        # --- determine best pair per candle
        dataframe["best_pair"] = momentum_df.idxmax(axis=1).reindex(dataframe.index)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = (
            (dataframe["best_pair"] == metadata["pair"]) &
            (dataframe["rebalance"])
        ).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = (
            (dataframe["best_pair"] != metadata["pair"]) &
            (dataframe["rebalance"])
        ).astype(int)
        return dataframe
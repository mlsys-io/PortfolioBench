from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import numpy as np
import logging

# Initialize the logger
logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

'''
Implements the Anti Correlation algorithm
'''
class AntiCorrelationPortfolio(IStrategy):
    position_adjustment_enable = True
    timeframe = '1d'
    minimal_roi = {} 
    stoploss = -1.00

    def __init__(self, config):
        super().__init__(config)
        self.pairs = self.config['exchange']['pair_whitelist']
        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs)}
        self.window = 100 # TODO: adjust based on timeframe of data
        self.new_values = np.zeros(len(self.pairs))
        self.last_update = None
        self.rebalance = False
        self.count = 0

    """
    Custom function to calculate the cross-correlation matrix
    y1: dictionary of pairs to their vectors (t-2w+1 to t-w)
    y2: dictionary of pairs to their vectors (t-w+1 to t)
    """
    def get_correlation_matrix(self, y1, y2):
        # Y1, Y2 are "window x num_pairs" matrices
        Y1 = np.array([y1[p].values for p in self.pairs]).T
        Y2 = np.array([y2[p].values for p in self.pairs]).T

        Y1_centred = Y1 - Y1.mean(axis=0)
        Y2_centred = Y2 - Y2.mean(axis=0)
        Y1_std = Y1.std(axis=0, ddof=1)
        Y2_std = Y2.std(axis=0, ddof=1)

        M = np.nan_to_num((Y1_centred.T @ Y2_centred) / ((self.window - 1) * np.outer(Y1_std, Y2_std)), nan=0.0, posinf=1.0, neginf=-1.0)
        logger.info(f"Correlation Matrix: \n{M}")
        return M


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        dataframe["log_relative"] = np.nan_to_num(np.log(dataframe["close"] / dataframe["close"].shift(1)))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 1  # Always signal entry if not in a trade
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self.count += 1
        self.rebalance = False
        if self.count == self.window:
            self.count = 0
            self.rebalance = True
        
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float, **kwargs):
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        relevant_data = df.loc[df["date"] <= current_time]
        if not self.rebalance or len(relevant_data) < 2*self.window:
            return 0

        if self.rebalance and self.last_update != current_time:
            logger.info(f"Rebalancing for {current_time}")
            self.last_update = current_time
            y1, y2 = {}, {}
            for pair in self.pairs:
                df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                relevant_data = df.loc[df["date"] <= current_time]
                y1[pair] = relevant_data["log_relative"].iloc[-2*self.window:-self.window]
                y2[pair] = relevant_data["log_relative"].iloc[-self.window:]

            corr_matrix = self.get_correlation_matrix(y1, y2)

            neg_auto_corr = np.minimum(0, np.diag(corr_matrix))
            raw_transfer = np.maximum(0, corr_matrix)

            adjustment = neg_auto_corr[:, None] + neg_auto_corr[None, :] 

            y2_bar = np.array([np.mean(y2[p]) for p in self.pairs])
            outperformance_mask = y2_bar[:, None] > y2_bar[None, :]

            raw_transfer = np.where(outperformance_mask & (raw_transfer > 0), raw_transfer - adjustment, 0)
            logger.info(f"Transfer value: \n{raw_transfer}")
            np.fill_diagonal(raw_transfer, 0)

            total_outflow = np.sum(raw_transfer, axis=1)
            normalized_transfer = raw_transfer / np.maximum(total_outflow, 1.0)[:, None]
            logger.info(f"Normalized transfer: \n{normalized_transfer}")

            # represents the amount we put up for rebalancing for each asset
            change_in_val = np.zeros(len(self.pairs))
            open_trades = Trade.get_open_trades()
            for t in open_trades:
                if t.pair in self.pair_to_idx:
                    pair_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    pair_rates = pair_df.loc[pair_df["date"] <= current_time].iloc[-self.window:]["close"]
                    change_in_rate = pair_rates.iloc[self.window-1]-pair_rates.iloc[0]
                    change_in_val[self.pair_to_idx[t.pair]] = t.amount * change_in_rate
            
            # ignore negative values since we cannot "transfer" negative money
            change_in_val = np.maximum(0, change_in_val)
            outflow = change_in_val * np.sum(normalized_transfer, axis=1)
            self.new_values = change_in_val @ normalized_transfer - outflow
            logger.info(f"Outflow       : {outflow}")
            logger.info(f"Change in val : {change_in_val}")
            logger.info(f"New values    : {self.new_values}")
        
        difference = self.new_values[self.pair_to_idx[trade.pair]]
        logger.info(f"PAIR: {trade.pair} | TIME: {current_time.date()} | CURRENT: {trade.amount * current_rate:.2f} | DIFF: {difference:.2f}")
        return difference
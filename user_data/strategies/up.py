from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

'''
Implements the Universal Portfolio algorithm
'''
class UniversalPortfolio(IStrategy):
    position_adjustment_enable = True
    timeframe = '1d'

    minimal_roi = {} 
    stoploss = -0.99

    def __init__(self, config):
        super().__init__(config)
        self.pairs = self.config['exchange']['pair_whitelist']
        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs)}
        self.n_samples = 10000
        self.portfolio_samples = self.generate_portfolio_samples(self.n_samples, len(self.pairs))
        self.cumulative_wealth = np.ones(self.n_samples)
        self.window = 100 # TODO: adjust based on timeframe of data
        self.next_weights = None
        self.last_update_time = None
        self.count = 0
        self.rebalance = False

    def generate_portfolio_samples(self, n_samples, n_assets):
        """
        Generates n_samples of weight vectors for n_assets.
        Weights sum to 1.0 and are uniformly distributed over the simplex.
        """
        # Use Dirichlet distribution with alpha=[1, 1, ..., 1]
        # This ensures every point on the simplex has an equal probability of being picked.
        alpha = np.ones(n_assets)
        samples = np.random.dirichlet(alpha, size=n_samples)
        
        return samples


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_indicators(dataframe, metadata)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = 1  # Always signal entry if not in a trade
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
    
    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        df_check = self.dp.get_pair_dataframe(self.pairs[0], self.timeframe)
        if df_check.empty:
            return
        
        relatives = []
        # logic to query and update the relative prices
        for pair in self.pairs:
            df = self.dp.get_pair_dataframe(pair, self.timeframe)
            if len(df) < self.window + 1:
                relatives.append(1.0)
                continue
            
            relative = df['close'].iloc[-1] / df['close'].iloc[-self.window-1]
            relatives.append(relative)

        relatives = np.array(relatives)
        performance = np.dot(self.portfolio_samples, relatives)
        self.cumulative_wealth = self.cumulative_wealth * performance
        self.next_weights = np.dot(self.cumulative_wealth, self.portfolio_samples) / np.sum(self.cumulative_wealth)
        self.last_update_time = current_time
        self.count += 1
        self.rebalance = False
        if self.count == self.window:
            self.rebalance = True
            self.count = 0
        
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float, **kwargs):
        
        # Get total account value that is tradeable (cash + existing positions)
        if self.rebalance:
            logger.info(f"Rebalancing for {current_time}")
            total_value = self.wallets.get_total_stake_amount()
            idx = self.pair_to_idx[trade.pair]
            target_weight = self.next_weights[idx]
            target_value = target_weight * total_value
        
            current_pos_value = trade.amount * current_rate

            return target_value - current_pos_value
        
        return 0
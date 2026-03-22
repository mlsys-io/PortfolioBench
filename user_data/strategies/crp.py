from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)
np.set_printoptions(precision=3, suppress=True, linewidth=150)

'''
Implements the Constant Rebalanced Portfolio
'''
class ConstantRebalancedPortfolio(IStrategy):
    position_adjustment_enable = True
    timeframe = '1w'

    minimal_roi = {} 
    stoploss = -0.99 

    def __init__(self, config):
        super().__init__(config)
        self.window = 100 # TODO: adjust based on timeframe of data
        self.count = 0
        self.rebalance = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return super().populate_indicators(dataframe, metadata)

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
        
        # Get total account value that is tradeable (cash + existing positions)
        if self.rebalance:
            logger.info(f"Rebalancing for {current_time}")
            total_value = self.wallets.get_total_stake_amount()
            target_value_per_coin = total_value / self.max_open_trades
            
            current_pos_value = trade.amount * current_rate

            return target_value_per_coin - current_pos_value

        return 0
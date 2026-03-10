import numpy as np
import pandas as pd
from scipy.optimize import minimize
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from datetime import datetime
from typing import Optional
import logging

# Initialize the logger
logger = logging.getLogger(__name__)

class ONS_Portfolio(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m' 
    stoploss = -1.00  # No stop loss
    max_entry_position_adjustment = -1 # Unlimited additional order for each open trade on top of the first entry order
    
    # Enable position adjustments for rebalancing
    position_adjustment_enable = True
    
    PARAMS = {
        "eta": 0.0,
        "beta": 1.0,
        "delta": 0.125
    }

    # Custom: Calculate daily target weight formula dynamically
    def calculate_ons_weights(self, price_data: pd.DataFrame) -> pd.DataFrame:
        target_pairs = price_data.columns.tolist()
        n_assets = len(target_pairs)
        n_rows = len(price_data)
        
        A = np.eye(n_assets)
        b = np.zeros(n_assets)
        p = np.ones(n_assets) / n_assets # Initialize p_0
        
        weights_history = np.zeros((n_rows, n_assets))
        
        prices = price_data.values
        r_matrix = np.vstack([np.ones(n_assets), prices[1:] / prices[:-1]]) # Calculate r_t
        
        # Iterate through the time period
        for t in range(n_rows):
            weights_history[t] = p
            
            r_t = r_matrix[t]
            
            portfolio_return = np.dot(p, r_t)
            if portfolio_return < 1e-8: portfolio_return = 1e-8
            grad = r_t / portfolio_return # Calculate gradient as in the paper's analysis
            
            A += np.outer(grad, grad) # Update hessian matrix --> Volatility adjustment
            b += (1 + 1.0/self.PARAMS['beta']) * grad # Get b_t --> Profit adjustment
            
            try:
                A_inv = np.linalg.pinv(A)
            except:
                A_inv = np.eye(n_assets)
                
            q = self.PARAMS['delta'] * A_inv.dot(b)
            
            # Get the projection of q
            p_next = self._project_simplex_A_norm(q, A, n_assets)
            
            if self.PARAMS['eta'] > 0:
                uniform = np.ones(n_assets) / n_assets
                p_next = (1 - self.PARAMS['eta']) * p_next + self.PARAMS['eta'] * uniform
            
            p = p_next

        # Store the weights of portfolio on days
        weights_df = pd.DataFrame(weights_history, index=price_data.index, columns=target_pairs)
        return weights_df

    # Custom: Solve for the projection step in the newton method
    def _project_simplex_A_norm(self, q, A, n):
        def objective(p):
            diff = q - p
            return diff.T @ A @ diff

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 0.95}) # Leave some extra cash to adjust portfolio
        bounds = tuple((0.0, 1.0) for _ in range(n))
        x0 = np.ones(n) / n # Start at uniform
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
        return res.x

    # Calculate target weight to be rebalanced to
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if not hasattr(self, 'ons_weights_cache'):
            self.ons_weights_cache = {}

        current_pair = metadata['pair']
        
        if self.dp:
            # Dynamically fetch the current whitelist provided to the bot
            target_pairs = self.dp.current_whitelist()
            price_dict = {}
            for pair in target_pairs:
                inf_df = self.dp.get_pair_dataframe(pair, self.timeframe)
                price_dict[pair] = inf_df['close']
            
            aligned_prices = pd.DataFrame(price_dict).dropna()
            
            if not aligned_prices.empty:
                # Cache key includes the number of pairs to prevent overlap between different test runs
                cache_key = str(aligned_prices.index[0]) + str(aligned_prices.index[-1]) + str(len(target_pairs))
                
                if cache_key not in self.ons_weights_cache:
                    logger.info(f"Calculating ONS weights for {len(aligned_prices)} candles across {len(target_pairs)} pairs...")
                    weights_df = self.calculate_ons_weights(aligned_prices) # Calculate portfolio weight
                    self.ons_weights_cache[cache_key] = weights_df
                
                weights = self.ons_weights_cache[cache_key]
                
                if current_pair in weights.columns:
                    dataframe['target_weight'] = dataframe.index.map(weights[current_pair])
                else:
                    dataframe['target_weight'] = 0.0
            else:
                dataframe['target_weight'] = 0.0
                
        return dataframe

    # ONS is a portfolio balancing that ensures non-zero for each target variable
    # Thus, entry is always 1 and exit is 0
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'enter_long'] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe

    # Update stake dynamically
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]
        
        target_weight = last_candle.get('target_weight', 0)
        
        # Get current portfolio
        if self.wallets:
            total_wallet = self.wallets.get_free(self.config['stake_currency'])
            open_trades = Trade.get_open_trades()
        else:
            total_wallet = self.config['dry_run_wallet']
            open_trades = []

        for t in open_trades:
            if t.pair == pair:
                total_wallet += t.amount * current_rate
            else:
                try:
                    pair_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    current_pair_rate = pair_df.loc[pair_df['date'] == current_time, 'close'].values[0]
                    total_wallet += t.amount * current_pair_rate
                except (IndexError, KeyError):
                    total_wallet += t.stake_amount 

        return total_wallet * target_weight

    # Helps to rebalance position
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              hp_value: Optional[float] = None, 
                              hp_present: Optional[float] = None,
                              **kwargs) -> Optional[float]:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if len(dataframe) > 0:
            last_candle = dataframe.iloc[-1]
            target_weight = last_candle['target_weight']
        else:
            return None

        # Get current portfolio balance
        if self.wallets:
            total_wallet = self.wallets.get_free(self.config['stake_currency'])
            open_trades = Trade.get_open_trades()
        else:
            total_wallet = self.config['dry_run_wallet']
            open_trades = []

        for t in open_trades:
            if t.pair == trade.pair:
                total_wallet += t.amount * current_rate
            else:
                try:
                    pair_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    current_pair_rate = pair_df.loc[pair_df['date'] == current_time, 'close'].values[0]
                    total_wallet += t.amount * current_pair_rate
                except (IndexError, KeyError):
                    total_wallet += t.stake_amount

        target_size = total_wallet * target_weight
        current_position_value = trade.amount * current_rate
        
        # Get adjustment needed
        diff = target_size - current_position_value

        # logger.info(f"PAIR: {trade.pair} | TARGET: {target_size:.2f} | CURRENT: {current_position_value:.2f} | DIFF: {diff:.2f}")

        # Only rebalance if the difference exceeds 2% of total portfolio value
        # to avoid excessive micro-adjustments that slow down backtesting
        if total_wallet > 0 and abs(diff) / total_wallet > 0.02:
            return diff

        return None
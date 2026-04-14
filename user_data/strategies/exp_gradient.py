import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)


class ExponentialGradientPortfolio(IStrategy):
    """
    Exponential Gradient (EG) online portfolio selection algorithm.

    Introduced by Helmbold et al. (1998) in "On-Line Portfolio Selection Using
    Multiplicative Updates". A multiplicative-weights algorithm that updates
    portfolio weights proportionally to recent asset returns. The learning rate
    (eta) controls how aggressively the portfolio shifts toward recent winners.
    Unlike ONS which uses second-order information, EG uses only first-order
    gradient updates, making it simpler and faster while still achieving
    logarithmic regret bounds.
    """

    INTERFACE_VERSION = 3
    timeframe = "5m"
    stoploss = -1.0

    process_only_new_candles = True
    startup_candle_count = 30
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    minimal_roi = {}

    # Learning rate: higher = more responsive to recent returns, lower = more stable
    ETA = 0.05

    def _compute_eg_weights(self, price_data: pd.DataFrame) -> pd.DataFrame:
        n_cols = len(price_data.columns)
        pairs = price_data.columns.tolist()
        n_rows = len(price_data)

        # Initialize with uniform weights
        weights_history = np.zeros((n_rows, n_cols))
        w = np.ones(n_cols) / n_cols

        for i in range(n_rows):
            weights_history[i] = w.copy()

            if i == 0:
                continue

            # Price relatives: x_t = price_t / price_{t-1}
            prev_prices = price_data.iloc[i - 1].values
            curr_prices = price_data.iloc[i].values

            # Avoid division by zero
            valid = prev_prices > 0
            if not valid.all():
                continue

            price_relatives = curr_prices / prev_prices

            # Portfolio return for normalization
            port_return = w @ price_relatives
            if port_return <= 0:
                continue

            # Multiplicative update: w_i *= exp(eta * x_i / (w . x))
            log_update = self.ETA * price_relatives / port_return
            w = w * np.exp(log_update)

            # Normalize to simplex (weights sum to 0.95 leaving cash buffer)
            total = w.sum()
            if total > 0:
                w = 0.95 * w / total
            else:
                w = np.ones(n_cols) / n_cols

        return pd.DataFrame(weights_history, index=price_data.index, columns=pairs)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not hasattr(self, 'eg_weights_cache'):
            self.eg_weights_cache = {}

        current_pair = metadata['pair']
        dataframe['target_weight'] = 0.0

        if self.dp:
            price_dict = {}
            for pair in self.dp.current_whitelist():
                inf_df = self.dp.get_pair_dataframe(pair, self.timeframe)
                price_dict[pair] = inf_df['close']

            aligned_prices = pd.DataFrame(price_dict).dropna()

            if not aligned_prices.empty:
                cache_key = (
                    str(aligned_prices.index[0])
                    + str(aligned_prices.index[-1])
                    + str(len(self.dp.current_whitelist()))
                )

                if cache_key not in self.eg_weights_cache:
                    logger.info(
                        f"Calculating EG weights for {len(aligned_prices)} "
                        f"candles across {len(price_dict)} pairs..."
                    )
                    weights_df = self._compute_eg_weights(aligned_prices)
                    self.eg_weights_cache[cache_key] = weights_df

                weights = self.eg_weights_cache[cache_key]

                if current_pair in weights.columns:
                    dataframe['target_weight'] = dataframe.index.map(weights[current_pair])
                else:
                    dataframe['target_weight'] = 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = (
            dataframe['target_weight'] > 0.001
        ).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe

    def _get_total_wallet(self, pair: str, current_time: datetime, current_rate: float) -> float:
        if self.wallets:
            total_wallet = self.wallets.get_free(self.config['stake_currency'])
            for t in Trade.get_open_trades():
                if t.pair == pair:
                    total_wallet += t.amount * current_rate
                else:
                    try:
                        pair_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                        rate = pair_df.loc[pair_df['date'] == current_time, 'close'].values[0]
                        total_wallet += t.amount * rate
                    except (IndexError, KeyError):
                        total_wallet += t.stake_amount
            return total_wallet
        else:
            return self.config['dry_run_wallet']

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float],
                            max_stake: float, leverage: float, entry_tag: Optional[str],
                            side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        target_weight = dataframe.iloc[-1].get('target_weight', 0.0)
        if not np.isfinite(target_weight) or target_weight <= 0:
            return proposed_stake

        total_wallet = self._get_total_wallet(pair, current_time, current_rate)
        return total_wallet * target_weight

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None

        target_weight = dataframe.iloc[-1].get('target_weight', 0.0)

        if not np.isfinite(target_weight) or target_weight <= 0:
            return None

        total_wallet = self._get_total_wallet(trade.pair, current_time, current_rate)
        target_size = target_weight * total_wallet
        current_position_value = trade.amount * current_rate

        diff = target_size - current_position_value

        # Only rebalance if difference exceeds 2% of portfolio
        if abs(diff) / total_wallet < 0.02:
            return None

        if abs(diff) < min_stake:
            return None

        if diff < 0:
            diff = max(diff, -trade.stake_amount)
            if abs(diff) < min_stake:
                return None

        return float(diff)

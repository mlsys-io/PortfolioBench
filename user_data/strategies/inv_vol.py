import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame

logger = logging.getLogger(__name__)


class InverseVolatilityPortfolio(IStrategy):
    """
    Inverse Volatility Portfolio strategy with stake allocation.
    - Computes daily returns over a lookback window.
    - Calculates inverse-volatility weights.
    - Rebalances monthly.
    """

    INTERFACE_VERSION = 3
    timeframe = "1d"
    stoploss = -1.0

    LOOKBACK = 30
    process_only_new_candles = True
    startup_candle_count = 30
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    minimal_roi = {}

    def _compute_ivp_weights(self, price_data: pd.DataFrame) -> pd.DataFrame:
        n_cols = len(price_data.columns)
        pairs = price_data.columns.tolist()
        n_rows = len(price_data)
        fallback = np.ones(n_cols) / n_cols

        weights_history = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            if i < self.LOOKBACK:
                weights_history[i] = fallback
                continue

            window = price_data.iloc[i - self.LOOKBACK:i]
            returns = window.pct_change().dropna()

            if returns.empty:
                weights_history[i] = fallback
                continue

            vol = returns.std(axis=0)
            vol = vol.replace(0, np.nan)
            inv_vol = 1 / vol
            inv_vol = inv_vol.fillna(0)

            if inv_vol.sum() > 0:
                weights_history[i] = inv_vol / inv_vol.sum()
            else:
                weights_history[i] = fallback

        return pd.DataFrame(weights_history, columns=pairs, index=price_data.index)


    def _get_first_trading_day(self, dataframe: DataFrame) -> pd.Series:
        """
        Returns a boolean Series that is True on the first available
        trading day of each month in the dataframe.
        """
        dates = pd.to_datetime(dataframe['date'])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
        else:
            dates = dates.dt.tz_localize(None)
        month_year = dates.dt.to_period('M')
        # Mark the first occurrence of each month
        first_trading_day = ~month_year.duplicated(keep='first')
        return first_trading_day


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not hasattr(self, 'ivp_weights_cache'):
            self.ivp_weights_cache = {}

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
                    str(aligned_prices.index[0]) +
                    str(aligned_prices.index[-1]) +
                    str(len(self.dp.current_whitelist()))
                )

                if cache_key not in self.ivp_weights_cache:
                    logger.info(f"Calculating IVP weights for {len(aligned_prices)} candles across {len(price_dict)} pairs...")
                    weights_df = self._compute_ivp_weights(aligned_prices)
                    self.ivp_weights_cache[cache_key] = weights_df

                weights = self.ivp_weights_cache[cache_key]

                if current_pair in weights.columns:
                    dataframe['target_weight'] = dataframe.index.map(weights[current_pair])
                else:
                    dataframe['target_weight'] = 0.0
        
        dataframe["rebalance"] = self._get_first_trading_day(dataframe)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = (
            (dataframe['target_weight'] > 0) &
            (dataframe['rebalance'] == True)
        ).astype(int)
        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (
            (dataframe['target_weight'] == 0) &
            (dataframe['rebalance'] == True)
        ).astype(int)
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        target_weight = dataframe.iloc[-1].get('target_weight', 0.0)
        if not np.isfinite(target_weight) or target_weight <= 0:
            return proposed_stake

        total_wallet = self._get_total_wallet(pair, current_time, current_rate)
        return total_wallet * target_weight

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
        else:
            total_wallet = self.config['dry_run_wallet']
        return total_wallet


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float, max_stake: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe.empty:
            return None
        
        if dataframe.iloc[-1].get('rebalance', False) == False:
            return None

        target_weight = dataframe.iloc[-1].get('target_weight', 0.0)

        if not np.isfinite(target_weight) or target_weight <= 0:
            return None

        total_wallet = self._get_total_wallet(trade.pair, current_time, current_rate)
        target_size = target_weight * total_wallet
        current_position_value = trade.amount * current_rate

        diff = target_size - current_position_value

        if abs(diff) < min_stake:
            return None

        if diff < 0:
            diff = max(diff, -trade.stake_amount)
            if abs(diff) < min_stake:
                return None

        return float(diff)

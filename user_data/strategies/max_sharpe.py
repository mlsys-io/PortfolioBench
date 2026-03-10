from freqtrade.persistence.trade_model import Trade
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


class MaxSharpePortfolio(IStrategy):
    """
    Maximum Sharpe Ratio (Markowitz Mean-Variance Optimization) portfolio strategy.

    Introduced by Harry Markowitz (1952) and extended by William Sharpe (1966).
    Finds the portfolio on the efficient frontier that maximizes the ratio of
    expected excess return to portfolio volatility over a rolling lookback window.
    Rebalances monthly on the first trading day of each month.
    """

    INTERFACE_VERSION = 3
    timeframe = "1d"
    stoploss = -1.0

    LOOKBACK = 60
    process_only_new_candles = True
    startup_candle_count = 60
    position_adjustment_enable = True
    max_entry_position_adjustment = -1

    minimal_roi = {}

    def _compute_max_sharpe_weights(self, price_data: pd.DataFrame) -> pd.DataFrame:
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

            if returns.empty or len(returns) < 2:
                weights_history[i] = fallback
                continue

            try:
                mu = returns.mean().values
                cov = returns.cov().values

                def neg_sharpe(w):
                    port_ret = w @ mu
                    port_vol = np.sqrt(w @ cov @ w)
                    if port_vol < 1e-10:
                        return 0.0
                    return -(port_ret / port_vol)

                constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
                bounds = [(0.0, 1.0)] * n_cols
                x0 = fallback.copy()

                result = minimize(
                    neg_sharpe, x0,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 200, "ftol": 1e-9},
                )

                if result.success:
                    w = result.x
                    w = np.maximum(w, 0)
                    total = w.sum()
                    weights_history[i] = w / total if total > 0 else fallback
                else:
                    weights_history[i] = fallback

            except Exception:
                weights_history[i] = fallback

        return pd.DataFrame(weights_history, index=price_data.index, columns=pairs)

    def _get_first_trading_day(self, dataframe: DataFrame) -> pd.Series:
        dates = pd.to_datetime(dataframe['date'])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_convert(None)
        else:
            dates = dates.dt.tz_localize(None)
        month_year = dates.dt.to_period('M')
        first_trading_day = ~month_year.duplicated(keep='first')
        return first_trading_day

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not hasattr(self, 'sharpe_weights_cache'):
            self.sharpe_weights_cache = {}

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

                if cache_key not in self.sharpe_weights_cache:
                    logger.info(
                        f"Calculating Max Sharpe weights for {len(aligned_prices)} "
                        f"candles across {len(price_dict)} pairs..."
                    )
                    weights_df = self._compute_max_sharpe_weights(aligned_prices)
                    self.sharpe_weights_cache[cache_key] = weights_df

                weights = self.sharpe_weights_cache[cache_key]

                if current_pair in weights.columns:
                    dataframe['target_weight'] = dataframe.index.map(weights[current_pair])
                else:
                    dataframe['target_weight'] = 0.0

        dataframe["rebalance"] = self._get_first_trading_day(dataframe)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'enter_long'] = (
            (dataframe['target_weight'] > 0)
            & (dataframe['rebalance'] == True)
        ).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = (
            (dataframe['target_weight'] == 0)
            & (dataframe['rebalance'] == True)
        ).astype(int)
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

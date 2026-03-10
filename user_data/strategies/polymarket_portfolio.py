"""Polymarket Event Portfolio Strategy — diversified prediction market portfolio.

Allocates capital across multiple Polymarket event contracts using
a modified Kelly criterion adapted for binary outcome contracts.

Key concepts:
- Each contract is a binary bet: pays $1 if YES, $0 if NO
- Expected value = estimated_prob * 1 - (1 - estimated_prob) * cost
- Kelly fraction = (p * b - q) / b  where b = (1/price - 1), p = estimated prob
- We use a fractional Kelly (1/4) for safety

The strategy estimates "fair probability" using momentum-adjusted price
and sizes positions based on perceived edge vs market price.
"""

import pandas as pd
from datetime import datetime
from typing import Optional
import logging

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class PolymarketPortfolio(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "4h"
    stoploss = -0.50
    max_entry_position_adjustment = -1
    position_adjustment_enable = True

    KELLY_FRACTION = 0.25  # Quarter-Kelly for safety
    REBALANCE_THRESHOLD = 0.03  # Rebalance if weight differs by >3%
    MIN_EDGE = 0.02  # Minimum edge (estimated prob - market price) to trade

    def _estimate_fair_prob(self, dataframe: pd.DataFrame) -> pd.Series:
        """Estimate 'fair' probability using smoothed momentum-adjusted price.

        Uses a slow EMA as the fair value estimate — if the market is at 0.40
        but the slow EMA suggests 0.45, there may be a buying opportunity.
        """
        close = dataframe["close"]
        # Blend of current price and slow moving average
        slow_ema = close.ewm(span=50, adjust=False).mean()
        fast_ema = close.ewm(span=10, adjust=False).mean()
        # Fair prob = weighted combination favoring the slow signal
        fair_prob = 0.6 * slow_ema + 0.4 * fast_ema
        return fair_prob.clip(0.02, 0.98)

    def _kelly_weight(self, fair_prob: float, market_price: float) -> float:
        """Calculate fractional Kelly bet size for a binary contract.

        For a YES contract at price `market_price`:
        - Payoff if win: (1 - market_price) / market_price  (the "odds")
        - Kelly = (p * b - q) / b
          where p = fair_prob, q = 1 - fair_prob, b = (1/market_price - 1)
        """
        if market_price <= 0.01 or market_price >= 0.99:
            return 0.0

        b = (1.0 / market_price) - 1.0  # decimal odds
        if b <= 0:
            return 0.0

        p = fair_prob
        q = 1.0 - p
        kelly = (p * b - q) / b

        # Only bet if there's positive edge
        if kelly <= 0:
            return 0.0

        # Fractional Kelly, capped
        return min(kelly * self.KELLY_FRACTION, 0.25)

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        if not hasattr(self, "_weight_cache"):
            self._weight_cache = {}

        # Core indicators
        close = dataframe["close"]
        dataframe["prob_ema_fast"] = close.ewm(span=10, adjust=False).mean()
        dataframe["prob_ema_slow"] = close.ewm(span=50, adjust=False).mean()
        dataframe["prob_momentum"] = close.diff(5).rolling(3).mean()

        # Fair probability estimate
        dataframe["fair_prob"] = self._estimate_fair_prob(dataframe)

        # Kelly weight for each candle
        kelly_weights = []
        for _, row in dataframe.iterrows():
            w = self._kelly_weight(row["fair_prob"], row["close"])
            kelly_weights.append(w)
        dataframe["kelly_weight"] = kelly_weights

        # Target weight: normalize across all pairs
        if self.dp:
            target_pairs = self.dp.current_whitelist()
            n_pairs = max(len(target_pairs), 1)

            # Simple: divide kelly weight proportionally, cap total at 0.95
            dataframe["target_weight"] = dataframe["kelly_weight"].clip(upper=0.95 / n_pairs)
        else:
            dataframe["target_weight"] = dataframe["kelly_weight"].clip(upper=0.20)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # Positive Kelly weight (edge exists)
                (dataframe["target_weight"] > 0.01)
                # Price in tradeable range
                & (dataframe["close"] > 0.05)
                & (dataframe["close"] < 0.95)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[
            (
                # No edge remaining
                (dataframe["target_weight"] < 0.005)
                # OR contract near resolution
                | (dataframe["close"] > 0.95)
                | (dataframe["close"] < 0.05)
            ),
            "exit_long",
        ] = 1
        return dataframe

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) == 0:
            return 0

        last_candle = dataframe.iloc[-1]
        target_weight = last_candle.get("target_weight", 0)

        # Calculate total portfolio value
        total_wallet = self._get_total_portfolio_value(pair, current_rate, current_time)

        return total_wallet * target_weight

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        hp_value: Optional[float] = None,
        hp_present: Optional[float] = None,
        **kwargs,
    ) -> Optional[float]:
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if len(dataframe) == 0:
            return None

        last_candle = dataframe.iloc[-1]
        target_weight = last_candle["target_weight"]

        total_wallet = self._get_total_portfolio_value(trade.pair, current_rate, current_time)
        target_size = total_wallet * target_weight
        current_position_value = trade.amount * current_rate

        diff = target_size - current_position_value

        if total_wallet > 0 and abs(diff) / total_wallet > self.REBALANCE_THRESHOLD:
            return diff

        return None

    def _get_total_portfolio_value(
        self, current_pair: str, current_rate: float, current_time: datetime
    ) -> float:
        """Calculate total portfolio value across all open positions + free wallet."""
        if self.wallets:
            total = self.wallets.get_free(self.config["stake_currency"])
            open_trades = Trade.get_open_trades()
        else:
            total = self.config.get("dry_run_wallet", 10000)
            open_trades = []

        for t in open_trades:
            if t.pair == current_pair:
                total += t.amount * current_rate
            else:
                try:
                    pair_df, _ = self.dp.get_analyzed_dataframe(t.pair, self.timeframe)
                    rate = pair_df.loc[pair_df["date"] == current_time, "close"].values[0]
                    total += t.amount * rate
                except (IndexError, KeyError):
                    total += t.stake_amount

        return total

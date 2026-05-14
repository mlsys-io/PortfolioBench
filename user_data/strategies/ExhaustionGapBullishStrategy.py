# pragma pylint: disable=missing-docstring, invalid-name
# flake8: noqa
# isort: skip_file

from datetime import datetime, timezone
from pandas import DataFrame
from typing import Optional

from freqtrade.strategy import IStrategy, Trade, timeframe_to_minutes
import talib.abstract as ta

class ExhaustionGapBullishStrategy(IStrategy):
    """
    Bullish Exhaustion Gap Strategy for 5m crypto (24/7) 

    Gap Idea:
      - Current candle opens below the previous candle's low by X% (a fast drop / dislocation),
        AND then closes bullish (close > open).
    Entry:
      - Enter long on the NEXT candle after the signal candle (more realistic).
    Exit:
      - Small mean-reversion exit: RSI gets hot OR price recovers above EMA.
      - Optional: takeprofit + time stop via custom_exit.
    """

    INTERFACE_VERSION = 3

    can_short: bool = False

    timeframe = "5m"
    process_only_new_candles = True

    minimal_roi = {}
    stoploss = -0.03
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    startup_candle_count: int = 200

    gap_open_below_prev_low_pct = 0.004   # 0.4%
    min_bull_body_pct = 0.001             # 0.1%

    ema_period = 50
    rsi_period = 14

    takeprofit_pct = 0.01         
    max_hold_candles = 24                # 24 * 5m[timeframe] = 120 minutes

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema"] = ta.EMA(dataframe, timeperiod=self.ema_period)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period)

        dataframe["low_prev"] = dataframe["low"].shift(1)
        dataframe["close_prev"] = dataframe["close"].shift(1)

        dataframe["open_below_prev_low_pct"] = (dataframe["low_prev"] - dataframe["open"]) / dataframe["low_prev"]

        # Bullish candle strength
        dataframe["bull_body_pct"] = (dataframe["close"] - dataframe["open"]) / dataframe["open"]

        dataframe["bull_gap_signal"] = (
            (dataframe["open_below_prev_low_pct"] > self.gap_open_below_prev_low_pct) &
            (dataframe["bull_body_pct"] > self.min_bull_body_pct) &
            (dataframe["close"] > dataframe["open"]) &
            (dataframe["volume"] > 0)
        ).astype("int")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Enter next candle after signal
        signal_prev = dataframe["bull_gap_signal"].shift(1).fillna(0).astype("int")

        dataframe.loc[
            (signal_prev == 1),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit when price recovers above EMA or RSI gets hot
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema"]) |
                (dataframe["rsi"] > 65)
            ),
            "exit_long",
        ] = 1

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        # Fixed TP
        if current_profit >= self.takeprofit_pct:
            return f"tp_{int(self.takeprofit_pct * 100)}pct"

        # Time stop
        tf_min = timeframe_to_minutes(self.timeframe)
        max_age_min = self.max_hold_candles * tf_min

        open_dt = trade.open_date_utc
        if open_dt.tzinfo is None:
            open_dt = open_dt.replace(tzinfo=timezone.utc)

        now_dt = current_time
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)

        age_minutes = (now_dt - open_dt).total_seconds() / 60.0
        if age_minutes >= max_age_min:
            return f"time_stop_{self.max_hold_candles}c"

        return None
    
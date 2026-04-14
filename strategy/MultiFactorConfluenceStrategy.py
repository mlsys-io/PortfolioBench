import numpy as np
import pandas as pd

"""
MultiFactorConfluenceStrategy — Multi-signal confluence scoring
===============================================================
Logic:
  Assigns a score (0–5) across five independent signals:
    1. MACD histogram turning positive
    2. RSI in 40-60 bullish zone (not overbought entry)
    3. Price above VWAP approximation (EMA of typical price × volume)
    4. Stochastic %K crossing above %D from oversold
    5. Volume above 20-bar average

  Entry  : score >= 3 (majority signals agree)
  Exit   : score <= 1 OR ROI hit
  Stop   : 6%

Suitable for: crypto, 4h timeframe — robust multi-asset design
"""

import talib.abstract as ta
from freqtrade.strategy import IntParameter, IStrategy
from pandas import DataFrame


class MultiFactorConfluenceStrategy(IStrategy):
    """AI-generated multi-factor strategy using signal scoring."""

    timeframe = "4h"
    minimal_roi = {"0": 0.18, "480": 0.10, "960": 0.05}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.06
    trailing_only_offset_is_reached = True
    can_short = False
    startup_candle_count = 50

    # Hyperopt-ready parameters
    entry_threshold = IntParameter(2, 5, default=3, space="buy")
    exit_threshold = IntParameter(0, 2, default=1, space="sell")
    rsi_bull_low = IntParameter(35, 50, default=40, space="buy")
    rsi_bull_high = IntParameter(55, 70, default=65, space="buy")
    stoch_oversold = IntParameter(15, 30, default=20, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # --- Factor 1: MACD ---
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macd_signal"] = macd["macdsignal"]
        dataframe["macd_hist"] = macd["macdhist"]
        dataframe["macd_hist_prev"] = dataframe["macd_hist"].shift(1)

        # --- Factor 2: RSI ---
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # --- Factor 3: VWAP approximation (EMA of typical price) ---
        dataframe["typical_price"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        # Weighted by volume: cumulative VWAP reset every 20 bars
        dataframe["tp_vol"] = dataframe["typical_price"] * dataframe["volume"]
        dataframe["vwap_approx"] = (
            dataframe["tp_vol"].rolling(20).sum()
            / dataframe["volume"].rolling(20).sum().replace(0, np.nan)
        )

        # --- Factor 4: Stochastic ---
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]
        dataframe["stoch_k_prev"] = dataframe["stoch_k"].shift(1)
        dataframe["stoch_d_prev"] = dataframe["stoch_d"].shift(1)

        # --- Factor 5: Volume ---
        dataframe["vol_ma20"] = dataframe["volume"].rolling(20).mean()

        # --- Compute confluence score (0–5) ---
        score = pd.Series(0, index=dataframe.index)

        # F1: MACD hist just turned positive
        score += ((dataframe["macd_hist"] > 0) & (dataframe["macd_hist_prev"] <= 0)).astype(int)
        # or already positive and growing
        score += ((dataframe["macd_hist"] > 0) & (dataframe["macd_hist"] > dataframe["macd_hist_prev"])).astype(int) * 0.5

        # F2: RSI in constructive zone
        score += (
            (dataframe["rsi"] > self.rsi_bull_low.value) & (dataframe["rsi"] < self.rsi_bull_high.value)
        ).astype(int)

        # F3: Price above VWAP
        score += (dataframe["close"] > dataframe["vwap_approx"]).astype(int)

        # F4: Stochastic cross from oversold
        score += (
            (dataframe["stoch_k"] > dataframe["stoch_d"])
            & (dataframe["stoch_k_prev"] <= dataframe["stoch_d_prev"])
            & (dataframe["stoch_k"] < 50)
        ).astype(int)

        # F5: Volume confirms move
        score += (dataframe["volume"] > dataframe["vol_ma20"]).astype(int)

        dataframe["confluence_score"] = score.round(1)

        # Exit score (bearish mirror)
        exit_score = pd.Series(0, index=dataframe.index)
        exit_score += ((dataframe["macd_hist"] < 0) & (dataframe["macd_hist_prev"] >= 0)).astype(int)
        exit_score += (dataframe["rsi"] > 70).astype(int)
        exit_score += (dataframe["close"] < dataframe["vwap_approx"]).astype(int)
        exit_score += (
            (dataframe["stoch_k"] < dataframe["stoch_d"])
            & (dataframe["stoch_k"] > 70)
        ).astype(int)
        dataframe["exit_score"] = exit_score

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["confluence_score"] >= self.entry_threshold.value)
                & (dataframe["volume"] > 0)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["exit_score"] >= self.exit_threshold.value + 2),
            "exit_long",
        ] = 1
        return dataframe

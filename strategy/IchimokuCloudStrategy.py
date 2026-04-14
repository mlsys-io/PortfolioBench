from datetime import datetime

import pandas as pd
from freqtrade.strategy import (
    IntParameter,
    IStrategy,
)
from pandas import DataFrame
from technical import qtpylib


class IchimokuCloudStrategy(IStrategy):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud) trend-following strategy.

    Developed by Goichi Hosoda in the 1930s and published in 1969.
    Enters long when price is above the cloud (Kumo), Tenkan-sen crosses
    above Kijun-sen (TK cross), and Chikou Span confirms the trend.
    """

    INTERFACE_VERSION = 3

    timeframe = "4h"

    can_short: bool = False

    minimal_roi = {}

    stoploss = -99

    trailing_stop = False

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 60

    # Ichimoku parameters (traditional values)
    tenkan_period = IntParameter(7, 12, default=9, space="buy")
    kijun_period = IntParameter(20, 30, default=26, space="buy")
    senkou_b_period = IntParameter(45, 60, default=52, space="buy")

    @staticmethod
    def _donchian_midline(dataframe: DataFrame, period: int) -> pd.Series:
        high = dataframe["high"].rolling(window=period).max()
        low = dataframe["low"].rolling(window=period).min()
        return (high + low) / 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_b_period.value

        # Tenkan-sen (Conversion Line)
        dataframe["tenkan_sen"] = self._donchian_midline(dataframe, tenkan)

        # Kijun-sen (Base Line)
        dataframe["kijun_sen"] = self._donchian_midline(dataframe, kijun)

        # Senkou Span A (Leading Span A) — shifted forward by kijun periods
        dataframe["senkou_span_a"] = (
            (dataframe["tenkan_sen"] + dataframe["kijun_sen"]) / 2
        ).shift(kijun)

        # Senkou Span B (Leading Span B) — shifted forward by kijun periods
        dataframe["senkou_span_b"] = self._donchian_midline(dataframe, senkou_b).shift(kijun)

        # Cloud top / bottom for easy comparison
        dataframe["cloud_top"] = dataframe[["senkou_span_a", "senkou_span_b"]].max(axis=1)
        dataframe["cloud_bottom"] = dataframe[["senkou_span_a", "senkou_span_b"]].min(axis=1)

        # Chikou Span (Lagging Span) — close shifted back by kijun periods
        dataframe["chikou_span"] = dataframe["close"].shift(-kijun)

        dataframe["mean-volume"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price above the cloud
                (dataframe["close"] > dataframe["cloud_top"])
                # Tenkan-sen crosses above Kijun-sen (bullish TK cross)
                & (qtpylib.crossed_above(dataframe["tenkan_sen"], dataframe["kijun_sen"]))
                # Senkou Span A above B (bullish cloud)
                & (dataframe["senkou_span_a"] > dataframe["senkou_span_b"])
                & (dataframe["mean-volume"] > 0.75)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Price drops below cloud
                (dataframe["close"] < dataframe["cloud_bottom"])
                # OR bearish TK cross
                | (qtpylib.crossed_below(dataframe["tenkan_sen"], dataframe["kijun_sen"]))
            ),
            "exit_long",
        ] = 1
        return dataframe

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float, rate: float,
        time_in_force: str, current_time: datetime, entry_tag: str | None,
        side: str, **kwargs,
    ) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_close = dataframe.iloc[-1]["close"]
        max_deviation = 0.01
        deviation = abs(rate - last_close) / last_close
        if deviation > max_deviation:
            return False
        return True

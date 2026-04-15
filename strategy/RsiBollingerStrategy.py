from datetime import datetime

import talib.abstract as ta
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
)
from pandas import DataFrame
from technical import qtpylib


class RsiBollingerStrategy(IStrategy):
    """
    RSI + Bollinger Bands mean-reversion strategy.

    Enters long when price touches the lower Bollinger Band and RSI is oversold.
    Exits when price reaches the upper Bollinger Band or RSI is overbought.
    A well-known approach described in Bollinger's "Bollinger on Bollinger Bands" (2001).
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"

    can_short: bool = False

    minimal_roi = {}

    stoploss = -99

    trailing_stop = False

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 30

    # Strategy parameters
    rsi_period = IntParameter(10, 20, default=14, space="buy")
    bb_period = IntParameter(15, 25, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")
    rsi_oversold = IntParameter(20, 40, default=30, space="buy")
    rsi_overbought = IntParameter(60, 80, default=70, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe),
            window=self.bb_period.value,
            stds=self.bb_std.value,
        )
        dataframe["bb_lower"] = bollinger["lower"]
        dataframe["bb_middle"] = bollinger["mid"]
        dataframe["bb_upper"] = bollinger["upper"]

        dataframe["mean-volume"] = dataframe["volume"].rolling(20).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] <= dataframe["bb_lower"])
                & (dataframe["rsi"] < self.rsi_oversold.value)
                & (dataframe["mean-volume"] > 0.75)
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["close"] >= dataframe["bb_upper"])
                | (dataframe["rsi"] > self.rsi_overbought.value)
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

# pragma pylint: disable=missing-docstring, invalid-name
# flake8: noqa
# isort: skip_file

from pandas import DataFrame

from freqtrade.strategy import IStrategy
import statsmodels.api as sm

class PairsTradingStrategy(IStrategy):
    """
    Strategy adapted from Gatev, Goetzmann & Rouwenhorst (2006)
    Implements cointegration-based pairs trading with z-score thresholds.
    """

    INTERFACE_VERSION = 3
    can_short: bool = False

    minimal_roi = {}
    stoploss = -0.05
    trailing_stop = False

    timeframe = "4h"
    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = 200

    # Define the two pairs to trade
    pair_a = "BTC/USDT"
    pair_b = "ETH/USDT"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Compute hedge ratio, spread, and z-score between two assets.
        """
        df_a = self.dp.get_pair_dataframe(self.pair_a, self.timeframe)
        df_b = self.dp.get_pair_dataframe(self.pair_b, self.timeframe)

        # Hedge ratio via linear regression
        model = sm.OLS(df_a['close'], sm.add_constant(df_b['close']))
        result = model.fit()
        beta = result.params[1]

        # Spread
        dataframe['spread'] = df_a['close'] - beta * df_b['close']

        # Z-score
        mean = dataframe['spread'].rolling(window=50).mean()
        std = dataframe['spread'].rolling(window=50).std()
        dataframe['zscore'] = (dataframe['spread'] - mean) / std

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry rules: trade when spread diverges significantly.
        """
        dataframe.loc[
            (dataframe['zscore'] > 2),
            'enter_short'
        ] = 1

        dataframe.loc[
            (dataframe['zscore'] < -2),
            'enter_long'
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit rules: close when spread mean-reverts.
        """
        dataframe.loc[
            (dataframe['zscore'].abs() < 0.5),
            ['exit_long', 'exit_short']
        ] = 1

        return dataframe


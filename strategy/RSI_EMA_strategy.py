
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta


class RSI_EMA_strategy(IStrategy):
    """
    Technicial indicator documentation: https://www.freqtrade.io/en/latest/strategy-customization/technical-indicators/
    github@: https://github.com/freqtrade/freqtrade-strategies

    Strategy description:

    Enter long when closing price is above the 1h EMA and RSI is oversold and then reclaims.
    RSI_reclaim: RSI goes back above 35 after being below 30 (oversold).
    Exit long when RSI is overbought or the price goes below the 1h EMA.

    Strategy Performance:

    Backtesting with data from 2025-02-02 00:00:00 up to 2025-10-29 00:00:00 (269 days) on 5m timeframe
    Pair: BTC/USDT
    Trades: 67
    Avg Profit: -0.45%
    Total Profit: -6.01%
    Sharpe: -1.18 
    """

    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #     "60":  0.01,
    #     "30":  0.03,
    #     "20":  0.04,
    #     "0":  0.05
    # }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # trailing stoploss
    trailing_stop = True

    # once profit hits 2%, set the trailing stop to 1%
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # run "populate_indicators" only for new candle
    process_only_new_candles = True

    # Experimental settings (configuration will overide these if set)
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_oversold"] = dataframe['rsi'] < 30
        dataframe["rsi_reclaim"] = (dataframe['rsi'] > 35) & (dataframe['rsi'].shift(1) <= 35)

        # 1h EMA
        # time period = 12 for 1h EMA on 5m timeframe
        ema_1h = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema_1h'] = ema_1h
        dataframe['uptrend'] = dataframe['close'] > dataframe['ema_1h']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['uptrend'] == True) & (dataframe['rsi_oversold'].shift(1) == True)
                & (dataframe['rsi_reclaim'] == True)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                ((dataframe['rsi'] > 60) & (dataframe['rsi'].shift(1) <= 60))
                | ((dataframe['uptrend'] == False))
            ),
            'exit_long'] = 1
        return dataframe
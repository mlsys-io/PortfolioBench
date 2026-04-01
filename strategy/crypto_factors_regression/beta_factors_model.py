
# --- Do not remove these libs ---
from math import tau
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

class beta_factors_model(IStrategy):

    INTERFACE_VERSION: int = 3
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {}

    can_short = False

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.15

    # Optimal timeframe for the strategy
    timeframe = '1w'

    # Minimum Candle count for indicator to populate
    startup_candle_count = 10

    # trailing stoploss
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    # Optional order type mapping
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Hyperoptable parameters
    buy_threshold = DecimalParameter(0.01, 0.09, decimals=3, default=0.07, space="buy")
    sell_threshold = DecimalParameter(0.01, 0.09, decimals=3, default=0.02, space="sell")

    # market cap dataframe
    btc_cap = None

    # load LSTM model
    MODEL_PATH = Path(__file__).resolve().parent / "beta_factors_model.joblib"
    _model = None

    def get_model(self):
        if self._model is None:
            self._model = joblib.load(self.MODEL_PATH)
        return self._model


    # load market cap data
    def load_market_cap_data(self):
        if self.btc_cap is not None:
            return self.btc_cap
        BASE_DIR = Path(__file__).resolve().parent
        PATH = BASE_DIR / "Bitcoin_marketcap.csv"
        data = pd.read_csv(PATH, sep=";")

        # modify timeClose to utc datetime
        data["timeClose"] = pd.to_datetime(data["timeClose"], utc=True)
        data["timeClose"] = data["timeClose"].dt.floor('D')

        # shift market cap to align with next day's prices
        data["marketCap_shifted"] = data["marketCap"].shift(1)

        self.btc_cap = data
        return data
    
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
    
    HOLD_DAYS = 7

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        # Exit after HOLD_DAYS regardless (time stop)
        if current_time - trade.open_date_utc >= timedelta(days=self.HOLD_DAYS):
            return "time_exit"
        return None


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.

        lookback timeframe: 15H
        Number of 5 min intervals = 15 x 60 / 5 = 180
        """

        # Load market cap data
        btc_cap = self.load_market_cap_data()

        # Merge market cap data into the dataframe
        dataframe["date"] = pd.to_datetime(dataframe["date"], utc=True)
        dataframe["week_end"] = (dataframe["date"].dt.floor('D') + pd.Timedelta(days=6)).dt.floor('D')
        sliced_btc_cap = btc_cap[["timeClose", "marketCap_shifted"]]
        dataframe = dataframe.merge(sliced_btc_cap, left_on="week_end", right_on="timeClose", how="left")

        # Design columns for ML model
        # define weekly returns based on closing prices
        dataframe['ret'] = dataframe['close'].pct_change()

        # Define CMKT Proxy as the weekly return of BTC-USD
        dataframe['cmkt'] = dataframe['ret']

        # Calculate CMOM(returns momentum) over a 2-week period
        dataframe['cmom'] = dataframe['ret'].rolling(window=2).sum()

        # Define mcap-cmkt interaction term
        dataframe['csize'] = np.log(dataframe['marketCap_shifted'])
        dataframe['csize_cmkt'] = dataframe['csize'] * dataframe['cmkt']

        # Higher Order terms
        dataframe['cmkt_2'] = dataframe['cmkt'] ** 2
        dataframe['cmom_3'] = dataframe['cmom'] ** 3

        # show dataframe with NaN values
        # print("Dataframe with NaN values:")
        # print(dataframe.head(20))

        # Replace NaN values with zero
        dataframe.fillna(0, inplace=True)

        # check the dataframe
        # print(dataframe.head())

        # Prepare data for prediction
        feature_cols = [
            'cmkt', 'cmom', 'csize', 'csize_cmkt', 'cmkt_2', 'cmom_3'
        ]
        X = dataframe[feature_cols].values
        # print(X.shape)

        # Predict
        pred_norm = self.get_model().predict(X)
        dataframe["pred_ret"] = np.nan

        # Align predictions to dataframe rows
        dataframe["pred_ret"] = pred_norm.flatten()
        # print("predicted returns:", pred_norm)

        dataframe["ml_signal"] = 0
        dataframe.loc[dataframe["pred_ret"] > 0.07, "ml_signal"] = 1
        dataframe.loc[dataframe["pred_ret"] < -0.07, "ml_signal"] = -1

        # ml signal = 0 if any features = 0 (to avoid trading on no info)
        dataframe.loc[
            (dataframe['cmkt'] == 0) | (dataframe['cmom'] == 0) | (dataframe['csize'] == 0) 
            | (dataframe['csize_cmkt'] == 0) | (dataframe['cmkt_2'] == 0) | (dataframe['cmom_3'] == 0), 
            'ml_signal'] = 0

        # print("first rows of dataframe with indicators:")
        # print(dataframe.head(50))

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        # Buy when predicted price increase is above threshold
        dataframe.loc[
            (
                (dataframe['ml_signal'] == 1) & 
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        # Sell when predicted price decrease is above threshold
        dataframe.loc[
            (
                (dataframe['ml_signal'] == -1) & 
                (dataframe['volume'] > 0) 
            ),
            'exit_long'] = 1
    
        
        return dataframe
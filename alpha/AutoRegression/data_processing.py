import numpy as np
import pandas as pd

def load_data(data_path: str = "./binance/BTC_USDT-5m.feather",
              start_date: str = "2021-01-01"):
    df = pd.read_feather(data_path)
    df = df[df["date"] > start_date]
    df["log_return"] = np.log(df['close'] / df['close'].shift(1))
    print(df.head())
    return df

def get_log_return_series():
    df = load_data()
    log_ret = df["log_return"].to_numpy()
    log_ret = log_ret[np.isfinite(log_ret)]
    return log_ret
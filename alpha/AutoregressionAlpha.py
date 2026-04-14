import numpy as np
import torch
from pandas import DataFrame

from alpha.AutoRegression.autoregression import build_train, trained_ar_model
from alpha.interface import IAlpha

AR_MODEL_PATH = "./ar_model.pth"
AR_LAG = 90

class AutoregressionAlpha(IAlpha):
    def process(self) -> DataFrame:
        df = self.dataframe.copy()
        
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        log_ret = df["log_return"].to_numpy()
        log_ret = log_ret[np.isfinite(log_ret)]
        
        if len(log_ret) <= AR_LAG:
            df["ar_pred"] = np.nan
            return df
        X, _ = build_train(log_ret, AR_LAG)
        # Load AR model
        ar_model = trained_ar_model
        ar_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            preds = ar_model(X_tensor).squeeze(1).cpu().numpy()
        # Align predictions with dataframe index
        ar_pred = np.full(df.shape[0], np.nan)
        ar_pred[AR_LAG+1:] = preds
        df["ar_pred"] = ar_pred
        return df
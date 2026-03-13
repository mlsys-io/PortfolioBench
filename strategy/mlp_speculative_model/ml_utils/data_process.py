import pandas as pd
from strategy.mlp_speculative_model.ml_utils.technical_analysis_tool import TecnicalAnalysis
from sklearn.preprocessing import StandardScaler
import numpy as np
features = [
    "z_score",
    "rsi",
    "boll",
    "ULTOSC",
    "pct_change",
    "zsVol",
    "PR_MA_Ratio_short",
    "MA_Ratio_short",
    "MA_Ratio",
    "PR_MA_Ratio",
    "DayOfWeek",
    "Month",
    "Hourly",
    "CDL2CROWS",
    "CDL3BLACKCROWS",
    "CDL3WHITESOLDIERS",
    "CDLABANDONEDBABY",
    "CDLBELTHOLD",
    "CDLCOUNTERATTACK",
    "CDLDARKCLOUDCOVER",
    "CDLDRAGONFLYDOJI",
    "CDLENGULFING",
    "CDLEVENINGDOJISTAR",
    "CDLEVENINGSTAR",
    "CDLGRAVESTONEDOJI",
    "CDLHANGINGMAN",
    "CDLHARAMICROSS",
    "CDLINVERTEDHAMMER",
    "CDLMARUBOZU",
    "CDLMORNINGDOJISTAR",
    "CDLMORNINGSTAR",
    "CDLPIERCING",
    "CDLRISEFALL3METHODS",
    "CDLSHOOTINGSTAR",
    "CDLSPINNINGTOP",
    "CDLUPSIDEGAP2CROWS",
]


class DataProcess:
    def __init__(self, path):
        data = pd.read_feather(path)
        self.df = DataProcess.process(data)

    scaler = StandardScaler()
    
    @staticmethod
    def process(data):
        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
        X = data[features].copy()

        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.apply(pd.to_numeric, errors="coerce") 
        X = X.fillna(0.0)

        Xs = DataProcess.scaler.fit_transform(X)
        return pd.DataFrame(Xs, columns=X.columns, index=X.index)




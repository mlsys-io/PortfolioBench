import pandas as pd
from strategy.mlp_speculative_model.ml_utils.technical_analysis_tool import TecnicalAnalysis

BUY = -1
HOLD = 0
SELL = 1


class LabelAlgorithm:
    def __init__(self, path):
        data = pd.read_feather(path)
        self.alpha = 0.038
        self.beta = 0.24
        self.b_window = 5
        self.f_window = 2

        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
        self.df = data

    def _find_alpha_beta(self):
        intl_hold = 0.85
        intl_buy_sell = 0.997
        self.alpha = self.df["pct_change"].abs().quantile(intl_hold)
        self.beta = self.df["pct_change"].abs().quantile(intl_buy_sell)

    def _find_bw_fw(self):
        pass
        # The original repo uses grid search for this one

    def label(self):
        self._find_alpha_beta()
        self._find_bw_fw()
        self.df["label"] = TecnicalAnalysis.assign_labels(
            self.df,
            b_window=self.b_window,
            f_window=self.f_window,
            alpha=self.alpha,
            beta=self.beta,
        )


if __name__ == "__main__":
    dp = LabelAlgorithm("./ml_model/data/BTC_USDT-4h.feather")

    dp.label()

    print(dp.df.head())
    print(dp.df.tail())

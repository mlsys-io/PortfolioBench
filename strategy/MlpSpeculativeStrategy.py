import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from freqtrade.strategy import IStrategy
from pandas import DataFrame

from strategy.mlp_speculative_model.ml_utils.data_process import DataProcess
from strategy.mlp_speculative_model.ml_utils.ensemble import sample_model
from strategy.mlp_speculative_model.ml_utils.technical_analysis_tool import TecnicalAnalysis


class MlpSpeculativeStrategy(IStrategy):
    can_short = False
    stoploss = -0.10
    
    minimal_roi = {"0" : 100}
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.0
    
    learner = sample_model
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = TecnicalAnalysis.compute_oscillators(dataframe)
        dataframe = TecnicalAnalysis.add_timely_data(dataframe)
        dataframe = TecnicalAnalysis.find_patterns(dataframe)
        dataframe = dataframe.dropna()
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # generate entry signals based on indicator values
        valid_idx = dataframe.dropna().index
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        if not valid_idx.empty:
            processed = DataProcess.process(dataframe.loc[valid_idx])
            # print("populate_exit_trend processed columns:", processed.columns.tolist())
            preds = self.learner.predict(processed)
            dataframe.loc[valid_idx, 'enter_long'] = (preds == 0).astype(int)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # generate exit signals based on indicator values
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        valid_idx = dataframe.dropna().index
        if not valid_idx.empty:
            processed = DataProcess.process(dataframe.loc[valid_idx])
            # print("populate_exit_trend processed columns:", processed.columns.tolist())
            preds = self.learner.predict(processed)
            dataframe.loc[valid_idx, 'exit_long'] = (preds == 2).astype(int)
        return dataframe

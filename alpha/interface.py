from abc import ABC, abstractmethod
from pandas import DataFrame

class IAlpha(ABC):
    def __init__(self, dataframe: DataFrame, metadata: dict = None):
        self.dataframe=dataframe
        self.metadata=metadata if metadata is not None else {}
        
    @abstractmethod
    def process(self) -> DataFrame:
        """
        This is to decouple the pipulation of indicator from IStrategy
        """
        pass

fwd_ret_timeframe = [1, 5, 10, 20, 90]
class AlphaEvaluator:
    def __init__(self, dataframe: DataFrame, alpha: type[IAlpha], metadata: dict = None):
        self.df = dataframe
        self.alpha = alpha(dataframe, metadata if metadata is not None else {})
        
    def evaluate_information_coefficient(self, alpha_names):
        df_processed = self.alpha.process()
        out = {}
        for a in alpha_names:
            for t in fwd_ret_timeframe:
                fwd_ret = df_processed['close'].pct_change().shift(-t)
                temp_df = DataFrame({'alpha': df_processed[a], 'fwd_ret': fwd_ret})
                temp_df = temp_df.dropna(subset=['alpha', 'fwd_ret'])
                ic = temp_df['alpha'].corr(temp_df['fwd_ret'], method='spearman')
                out[(a, t)] = ic
        return out
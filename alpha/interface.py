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
    def __init__(self, dataframe: DataFrame, alpha: IAlpha):
        self.df = dataframe
        self.alpha = alpha(dataframe)
        
    def evaluate_information_coefficient(self, alpha_names):
        self.df = self.alpha.process()
        out = {}
        for a in alpha_names:
            for t in fwd_ret_timeframe:
                self.df['fwd_ret'] = self.df['close'].pct_change(periods=t).shift(-t)
                self.df = self.df.dropna(subset=[a, 'fwd_ret'])
                ic = self.df['alpha'].corr(self.df['fwd_ret'], method='spearman')
                out[(a, t)] = ic
        return out
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
        
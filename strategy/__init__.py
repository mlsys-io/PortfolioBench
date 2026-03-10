from strategy.EmaCrossStrategy import EmaCrossStrategy
from strategy.MacdAdxStrategy import MacdAdxStrategy
from strategy.MlpSpeculativeStrategy import MlpSpeculativeStrategy
from strategy.RsiBollingerStrategy import RsiBollingerStrategy
from strategy.IchimokuCloudStrategy import IchimokuCloudStrategy
from strategy.StochasticCciStrategy import StochasticCciStrategy
from strategy.PolymarketMomentumStrategy import PolymarketMomentumStrategy
from strategy.PolymarketMeanReversionStrategy import PolymarketMeanReversionStrategy
strategy_list = [
    EmaCrossStrategy.__name__,
    MacdAdxStrategy.__name__,
    MlpSpeculativeStrategy.__name__,
    RsiBollingerStrategy.__name__,
    IchimokuCloudStrategy.__name__,
    StochasticCciStrategy.__name__,
    PolymarketMomentumStrategy.__name__,
    PolymarketMeanReversionStrategy.__name__,
]

if __name__ == "__main__":
    print(strategy_list)

from strategy.EmaCrossStrategy import EmaCrossStrategy
from strategy.IchimokuCloudStrategy import IchimokuCloudStrategy
from strategy.MacdAdxStrategy import MacdAdxStrategy
from strategy.MlpSpeculativeStrategy import MlpSpeculativeStrategy
from strategy.PolymarketMeanReversionStrategy import PolymarketMeanReversionStrategy
from strategy.PolymarketMomentumStrategy import PolymarketMomentumStrategy
from strategy.RsiBollingerStrategy import RsiBollingerStrategy
from strategy.StochasticCciStrategy import StochasticCciStrategy

strategy_list = [
    EmaCrossStrategy.__name__,
    MacdAdxStrategy.__name__,
    RsiBollingerStrategy.__name__,
    IchimokuCloudStrategy.__name__,
    StochasticCciStrategy.__name__,
    PolymarketMomentumStrategy.__name__,
    PolymarketMeanReversionStrategy.__name__,
]
if MlpSpeculativeStrategy is not None:
    strategy_list.append(MlpSpeculativeStrategy.__name__)

if __name__ == "__main__":
    print(strategy_list)

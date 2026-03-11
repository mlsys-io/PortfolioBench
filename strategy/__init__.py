from strategy.EmaCrossStrategy import EmaCrossStrategy
from strategy.MacdAdxStrategy import MacdAdxStrategy
from strategy.RsiBollingerStrategy import RsiBollingerStrategy
from strategy.IchimokuCloudStrategy import IchimokuCloudStrategy
from strategy.StochasticCciStrategy import StochasticCciStrategy
from strategy.PolymarketMomentumStrategy import PolymarketMomentumStrategy
from strategy.PolymarketMeanReversionStrategy import PolymarketMeanReversionStrategy
from strategy.MlpSpeculativeStrategy import MlpSpeculativeStrategy

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

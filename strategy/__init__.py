from strategy.EmaCrossStrategy import EmaCrossStrategy
from strategy.MacdAdxStrategy import MacdAdxStrategy
from strategy.ONS import ONS_Portfolio
from strategy.MlpSpeculativeStrategy import MlpSpeculativeStrategy
strategy_list = [
    EmaCrossStrategy.__name__,
    MacdAdxStrategy.__name__, 
    MlpSpeculativeStrategy.__name__
]

if __name__ == "__main__":
    print(strategy_list)
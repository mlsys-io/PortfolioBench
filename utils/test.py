from freqtrade.commands.optimize_commands import setup_optimize_configuration
from freqtrade.enums import RunMode
from freqtrade.optimize.backtesting import Backtesting

from strategy import strategy_list


def run_backtest(strategy):
    print("=" * 60)
    print(f"Test strategy {strategy}\n")

    args = {
        "config": ["./user_data/config.json"],
        "strategy": str(strategy),
        "timerange": "20250101-20250108",
        "timeframe": "5m",
        "strategy_path": "./strategy",
        "pairs": [
            "BTC/USDT",
            "ETH/USDT",
            # "SOL/USDT",
            # "XRP/USDT",
            # "MSFT/USDT"
        ],
    }

    config = setup_optimize_configuration(args, RunMode.BACKTEST)

    backtesting = Backtesting(config)
    backtesting.start()
    if backtesting.exchange:
        backtesting.exchange.close()

def main():
    for s in strategy_list:
        run_backtest(s)

if __name__ == "__main__":
    main() 
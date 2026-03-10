# PortfolioBench

A **multi-asset portfolio benchmarking framework** built as a wrapper around [freqtrade](https://github.com/freqtrade/freqtrade). PortfolioBench extends freqtrade's cryptocurrency-focused backtesting engine to support **US equities, global market indices, and mixed-asset portfolios**, while adding pluggable alpha-factor interfaces, academic portfolio-optimization algorithms, and a standalone portfolio-construction pipeline.

## Key Capabilities

- **Multi-asset backtesting** — Crypto, US stocks (~100 tickers), and global indices (DJI, S&P 500, FTSE, Nikkei, etc.)
- **Portfolio algorithms** — ONS (Online Newton Step), Minimum Variance, Inverse Volatility, Best Single Asset (Momentum Rotation)
- **Trading strategies** — EMA Crossover, MACD+ADX
- **Alpha factor abstraction** — Decoupled indicator computation via `IAlpha` interface
- **Blended portfolio construction** — Standalone pipeline combining ONS + EMA + Equal-Weight
- **119 instruments** across 3 timeframes (5m, 4h, 1d) with pre-downloaded OHLCV data

## Repository Layout

```
PortfolioBench/
├── freqtrade/                  # Vendored freqtrade (unmodified)
│   └── exchange/
│       └── portfoliobench.py   # Custom exchange subclass (extends Binance)
├── alpha/                      # Pluggable alpha-factor system
│   ├── interface.py            # IAlpha abstract base class
│   ├── SimpleEmaFactors.py     # EMA fast/slow/exit + rolling mean volume
│   └── PolymarketFactors.py    # Polymarket prediction-market factors
├── strategy/                   # Freqtrade IStrategy implementations
│   ├── EmaCrossStrategy.py     # EMA crossover entry/exit strategy
│   ├── MacdAdxStrategy.py      # MACD + ADX trend-confirmation strategy
│   ├── IchimokuCloudStrategy.py        # Ichimoku Cloud strategy
│   ├── RsiBollingerStrategy.py         # RSI + Bollinger Bands strategy
│   ├── StochasticCciStrategy.py        # Stochastic + CCI strategy
│   ├── MlpSpeculativeStrategy.py       # MLP-based speculative strategy
│   ├── PolymarketMeanReversionStrategy.py  # Polymarket mean reversion
│   └── PolymarketMomentumStrategy.py       # Polymarket momentum
├── portfolio/                  # Standalone portfolio pipeline
│   └── PortfolioManagement.py  # 7-step: load → alpha → signals → ONS → blend → backtest → metrics
├── dataset/                    # Data management (stub)
│   └── main.py
├── user_data/
│   ├── config.json             # Backtesting configuration
│   ├── strategies/             # Portfolio-optimization strategies
│   │   ├── ONS.py              # Online Newton Step rebalancing
│   │   ├── inv_vol.py          # Inverse Volatility allocation
│   │   ├── min_var.py          # Minimum Variance allocation
│   │   ├── best_single_asset.py # Momentum rotation (winner-takes-all)
│   │   ├── exp_gradient.py     # Exponential Gradient allocation
│   │   ├── max_sharpe.py       # Maximum Sharpe Ratio optimization
│   │   ├── risk_parity.py      # Risk Parity allocation
│   │   └── polymarket_portfolio.py # Polymarket portfolio strategy
│   └── data/binance/           # Pre-downloaded OHLCV data (357 feather files)
└── utils/
    ├── backtest_script.bash    # Simple CLI backtest launcher
    ├── backtest_tests.bash     # Comprehensive test harness
    ├── backtest_polymarket.bash        # Polymarket backtesting script
    ├── download_polymarket_data.py     # Polymarket data downloader
    ├── generate_polymarket_test_data.py # Polymarket test data generator
    ├── generate_test_data.py           # General test data generator
    └── test.py                         # Test runner
```

## Quick Start

### Backtest a Trading Strategy

```bash
freqtrade backtesting \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --timeframe 4h \
    --timerange 20250501-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT XRP/USDT
```

### Backtest a Portfolio Strategy

```bash
freqtrade backtesting \
    --strategy ONS_Portfolio \
    --strategy-path ./user_data/strategies \
    --timeframe 5m \
    --timerange 20260101-20260108 \
    --pairs BTC/USDT ETH/USDT AAPL/USDT MSFT/USDT DJI/USDT \
    --dry-run-wallet 1000000
```

### Run the Standalone Portfolio Pipeline

```bash
python -m portfolio.PortfolioManagement
```

This runs a 7-step pipeline (load data, generate indicators, compute signals, run ONS, set equal-weight, blend strategies, backtest) with default parameters:
- Pairs: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, MSFT/USDT
- Blend: 34% equal-weight + 33% ONS + 33% EMA
- Initial capital: $10,000

### Run the Full Test Suite

```bash
bash utils/backtest_tests.bash
```

Runs 4 asset categories (crypto, stocks, indices, mixed) across 3 timeframes = 12 test configurations.

## Trading Strategies

| Strategy | Type | Entry Signal | Exit Signal |
|----------|------|-------------|-------------|
| **EmaCrossStrategy** | Trend-following | EMA fast crosses above EMA slow + volume filter | EMA exit crosses below EMA fast |
| **MacdAdxStrategy** | Trend-confirmation | MACD > signal + ADX > 25 + volume filter | MACD < signal + ADX > 25 |
| **IchimokuCloudStrategy** | Trend-following | Ichimoku Cloud signals | Ichimoku Cloud reversal |
| **RsiBollingerStrategy** | Mean-reversion | RSI + Bollinger Bands | RSI + Bollinger Bands reversal |
| **StochasticCciStrategy** | Oscillator-based | Stochastic + CCI signals | Stochastic + CCI reversal |
| **MlpSpeculativeStrategy** | ML-based | MLP model predictions | MLP model predictions |
| **PolymarketMeanReversionStrategy** | Mean-reversion | Polymarket mean reversion signals | Reversion exit |
| **PolymarketMomentumStrategy** | Momentum | Polymarket momentum signals | Momentum exit |

## Portfolio Algorithms

| Strategy | Allocation Method | Rebalance Frequency | Positions |
|----------|------------------|---------------------|-----------|
| **ONS** | Online convex optimization (Newton Step) | Per-candle | All pairs |
| **Inverse Volatility** | Weight proportional to 1/volatility | Monthly | All pairs |
| **Minimum Variance** | Inverse-covariance weighting | Monthly | All pairs |
| **Best Single Asset** | Momentum rotation (winner-takes-all) | Monthly | 1 pair |
| **Exponential Gradient** | Multiplicative weight update | Per-candle | All pairs |
| **Maximum Sharpe** | Sharpe ratio optimization | Monthly | All pairs |
| **Risk Parity** | Equal risk contribution | Monthly | All pairs |
| **Polymarket Portfolio** | Prediction-market weighted | Monthly | All pairs |

## Asset Universe

- **Cryptocurrencies (10)**: BTC, ETH, SOL, XRP, DOGE, BNB, ADA, TRX, STETH, BCH
- **US Stocks (~100)**: AAPL, MSFT, NVDA, GOOG, AMZN, META, TSLA, JPM, and more (roughly S&P 100)
- **Global Indices (9)**: DJI, GSPC (S&P 500), IXIC (Nasdaq), RUT (Russell 2000), FTSE, N225 (Nikkei), HSI (Hang Seng), STOXX50E, VIX

All data is stored as feather files in `user_data/data/binance/` using the naming convention `{TICKER}_USDT-{timeframe}.feather`.

## Benchmarking

PortfolioBench supports systematic benchmarking across multiple dimensions:

| Dimension | Values |
|-----------|--------|
| Strategies | EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative, ONS, InverseVol, MinVar, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity |
| Asset classes | Crypto, US Stocks, Global Indices, Mixed |
| Timeframes | 5m, 4h, 1d |

This produces up to **156 benchmark configurations** (13 strategies x 4 asset classes x 3 timeframes).

**Metrics**: Total return, annualized return, Sharpe ratio, max drawdown, number of trades, win rate, profit factor.

## Extending PortfolioBench

### Adding a New Alpha Factor

```python
# alpha/MyAlpha.py
from alpha.interface import IAlpha

class MyAlpha(IAlpha):
    def process(self):
        self.dataframe["my_indicator"] = ...
        return self.dataframe
```

### Adding a New Trading Strategy

Implement `IStrategy` in `strategy/` with `populate_indicators`, `populate_entry_trend`, and `populate_exit_trend`.

### Adding a New Portfolio Algorithm

Implement `IStrategy` with `position_adjustment_enable = True` in `user_data/strategies/`. Use `custom_stake_amount()` and `adjust_trade_position()` for rebalancing.

### Adding New Assets

1. Prepare OHLCV data as feather files with columns: `date, open, high, low, close, volume`
2. Name files as `{TICKER}_USDT-{timeframe}.feather`
3. Place in `user_data/data/binance/`
4. Add pairs to the config whitelist or pass via `--pairs`

The `Portfoliobench` exchange subclass will auto-inject synthetic market entries for non-crypto tickers.

## Hyperparameter Optimization

```bash
freqtrade hyperopt \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --timeframe 4h \
    --timerange 20250101-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT \
    --epochs 100
```

## How It Works

PortfolioBench achieves multi-asset support with **minimal invasiveness**: a clean `Portfoliobench` exchange subclass (`freqtrade/exchange/portfoliobench.py`) extends Binance to handle non-crypto tickers, while the vendored `freqtrade/exchange/exchange.py` remains **unmodified**. All new functionality is added through freqtrade's existing extension points (strategies, callbacks, data format).

To use the custom exchange, set `"exchange": {"name": "portfoliobench"}` in your config.

## Limitations

- Long-only strategies (no short selling)
- Zero transaction costs for stocks (unrealistic for live trading)
- Backtesting only (exchange hacks would fail in live trading)
- All data must use Binance naming convention

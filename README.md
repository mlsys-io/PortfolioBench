# PortfolioBench

A multi-asset portfolio benchmarking framework for backtesting trading strategies and portfolio allocation algorithms across **cryptocurrencies**, **US equities**, **global indices**, and **event prediction markets** (Polymarket).

PortfolioBench ships with 119 instruments, 16 strategies, 5 alpha factors, and a full benchmarking suite — so you can evaluate portfolio ideas out of the box.

---

## What You Can Do

- **Backtest trading strategies** (EMA crossover, MACD, Ichimoku, RSI+Bollinger, MLP, and more) on any supported asset
- **Compare portfolio algorithms** (Online Newton Step, Risk Parity, Max Sharpe, Minimum Variance, etc.) head-to-head
- **Trade prediction markets** — backtest strategies on Polymarket binary outcome contracts (YES/NO shares priced $0–$1)
- **Mix asset classes** — build and test portfolios that span crypto, stocks, indices, and event contracts in a single run
- **Benchmark everything** — run all 16 strategies across 4 asset classes and 3 timeframes with one command

---

## Data Setup

OHLCV data is hosted on Google Drive. Use the `portbench` CLI to download it:

```bash
pip install gdown
portbench download-data --exchange portfoliobench   # crypto + US stocks + global indices
portbench download-data --exchange polymarket        # Polymarket prediction-market contracts
```

This downloads feather files into the data directory (`user_data/data/portfoliobench/` or `user_data/data/polymarket/`).

Alternatively, generate synthetic data for testing without downloading:

```bash
python utils/generate_test_data.py
```

---

## Quick Start

### 1. Backtest a Trading Strategy

```bash
portbench backtesting \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --timeframe 4h \
    --timerange 20250501-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT XRP/USDT
```

### 2. Backtest a Portfolio Algorithm

```bash
portbench backtesting \
    --strategy ONS_Portfolio \
    --strategy-path ./user_data/strategies \
    --timeframe 5m \
    --timerange 20260101-20260108 \
    --pairs BTC/USDT ETH/USDT AAPL/USD MSFT/USD DJI/USD \
    --dry-run-wallet 1000000
```

### 3. Run the Standalone Portfolio Pipeline

```bash
python -m portfolio.PortfolioManagement
```

Runs a 7-step pipeline: load data, generate indicators, compute signals, run ONS, set equal-weight, blend strategies, and backtest. Defaults to a 34% equal-weight + 33% ONS + 33% EMA blend on BTC, ETH, SOL, XRP, and MSFT.

### 4. Backtest a Polymarket Strategy

```bash
portbench backtesting \
    --strategy PolymarketMomentumStrategy \
    --strategy-path ./strategy \
    --timeframe 5m \
    --timerange 20260101-20260108 \
    --pairs TRUMP-WIN-YES/USDT ETH-10K-NO/USDT
```

Polymarket contracts are binary outcomes priced between $0 and $1. Each event has complementary YES/NO contracts that settle at exactly $0 or $1.

---

## Run the Full Benchmark

The benchmark suite tests every layer of the framework: data integrity, alpha factors, the portfolio pipeline, and all strategy backtests.

### Full run (all strategies, all assets, all timeframes)

```bash
portbench benchmark
```

### Quick smoke test

```bash
portbench benchmark --quick
```

### Only trading strategies or only portfolio strategies

```bash
portbench benchmark --trading-only
portbench benchmark --portfolio-only
```

### Export results to JSON

```bash
portbench benchmark --export report.json
```

### Advanced benchmark with filters

```bash
portbench benchmark-all --timeframes 5m 4h --categories crypto stocks --json-output results.json
```

### Benchmark Matrix

| Dimension      | Options |
|----------------|---------|
| **Trading strategies** (8) | EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative, PolymarketMeanReversion, PolymarketMomentum |
| **Portfolio algorithms** (8) | ONS, InverseVol, MinVar, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity, PolymarketPortfolio |
| **Asset classes** (4) | Crypto, US Stocks, Global Indices, Mixed |
| **Timeframes** (3) | 5m, 4h, 1d |

This gives up to **192 benchmark configurations** (16 strategies x 4 asset classes x 3 timeframes).

**Reported metrics**: total return, annualized return, Sharpe ratio, max drawdown, number of trades, win rate, profit factor.

---

## Alpha Factors

Alpha factors are pluggable indicator modules that enrich OHLCV data with derived signals. They implement the `IAlpha` interface and can be composed into trading strategies.

| Alpha | Indicators | Use Case |
|-------|-----------|----------|
| **EmaAlpha** | EMA fast/slow/exit, mean volume | Trend-following crossover signals |
| **RsiAlpha** | RSI, RSI signal line, overbought/oversold flags | Mean-reversion and momentum timing |
| **MacdAlpha** | MACD line, signal line, histogram, histogram direction | Trend-confirmation and momentum |
| **BollingerAlpha** | Upper/middle/lower bands, bandwidth, %B | Volatility breakouts and mean-reversion |
| **PolymarketAlpha** | Probability momentum, RSI, z-score, volume surge, resolution proximity | Prediction market contract analysis |

---

## Trading Strategies

| Strategy | Type | Entry Signal | Exit Signal |
|----------|------|-------------|-------------|
| **EmaCross** | Trend-following | EMA fast crosses above EMA slow + volume filter | EMA exit crosses below EMA fast |
| **MacdAdx** | Trend-confirmation | MACD > signal + ADX > 25 + volume filter | MACD < signal + ADX > 25 |
| **IchimokuCloud** | Trend-following | Ichimoku Cloud signals | Ichimoku Cloud reversal |
| **RsiBollinger** | Mean-reversion | RSI + Bollinger Bands | RSI + Bollinger Bands reversal |
| **StochasticCci** | Oscillator-based | Stochastic + CCI signals | Stochastic + CCI reversal |
| **MlpSpeculative** | ML-based | MLP model predictions | MLP model predictions |
| **PolymarketMeanReversion** | Mean-reversion | Polymarket contract reversion signals | Reversion exit |
| **PolymarketMomentum** | Momentum | Polymarket contract momentum signals | Momentum exit |

## Portfolio Algorithms

| Algorithm | Method | Rebalance Frequency |
|-----------|--------|---------------------|
| **ONS** | Online convex optimization (Newton Step) | Per-candle |
| **Inverse Volatility** | Weight proportional to 1/volatility | Monthly |
| **Minimum Variance** | Inverse-covariance weighting | Monthly |
| **Best Single Asset** | Momentum rotation (winner-takes-all) | Monthly |
| **Exponential Gradient** | Multiplicative weight update | Per-candle |
| **Maximum Sharpe** | Sharpe ratio optimization | Monthly |
| **Risk Parity** | Equal risk contribution | Monthly |
| **Polymarket Portfolio** | Prediction-market weighted allocation | Monthly |

---

## Asset Universe

| Class | Count | Examples |
|-------|-------|---------|
| **Crypto** | 10 | BTC, ETH, SOL, XRP, DOGE, BNB, ADA, TRX, STETH, BCH |
| **US Stocks** | ~100 | AAPL, MSFT, NVDA, GOOG, AMZN, META, TSLA, JPM, ... |
| **Global Indices** | 9 | DJI, S&P 500, Nasdaq, Russell 2000, FTSE, Nikkei, Hang Seng, STOXX 50, VIX |
| **Prediction Markets** | varies | Polymarket binary outcome contracts (YES/NO) |

All instruments are available at 3 timeframes (5m, 4h, 1d) as pre-downloaded OHLCV feather files — **357 files total**.

---

## Event Prediction Markets (Polymarket)

PortfolioBench supports backtesting on **Polymarket**, a decentralized prediction market where binary outcome contracts trade between $0 and $1.

**How it works**:
- Each event (e.g., "Will ETH reach $10K?") has two contracts: YES and NO
- Contract prices represent the market's implied probability of the outcome
- YES + NO prices sum to approximately $1
- Contracts settle at exactly $0 or $1 when the event resolves
- Profit = (settlement price - entry price) x shares

**Pair convention**: `{EVENT_SLUG}-{YES|NO}/USDT` (e.g., `TRUMP-WIN-YES/USDT`)

**Included strategies**:
- `PolymarketMomentumStrategy` — trades momentum in contract prices
- `PolymarketMeanReversionStrategy` — trades mean reversion in contract prices
- `PolymarketPortfolio` — portfolio allocation across multiple prediction market contracts

**Config**: use `user_data/config_polymarket.json` or run:
```bash
bash utils/backtest_polymarket.bash
```

---

## Hyperparameter Optimization

```bash
portbench hyperopt \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --timeframe 4h \
    --timerange 20250101-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT \
    --epochs 100
```

---

## Extending PortfolioBench

### Add a new alpha factor

```python
# alpha/MyAlpha.py
from alpha.interface import IAlpha

class MyAlpha(IAlpha):
    def process(self):
        self.dataframe["my_indicator"] = ...
        return self.dataframe
```

### Add a new trading strategy

Implement `IStrategy` in `strategy/` with `populate_indicators`, `populate_entry_trend`, and `populate_exit_trend`.

### Add a new portfolio algorithm

Implement `IStrategy` with `position_adjustment_enable = True` in `user_data/strategies/`. Use `custom_stake_amount()` and `adjust_trade_position()` for rebalancing.

### Add new assets

1. Prepare OHLCV data as feather files with columns: `date, open, high, low, close, volume`
2. Name files: `{TICKER}_USDT-{timeframe}.feather` (crypto) or `{TICKER}_USD-{timeframe}.feather` (stocks/indices)
3. Place in `user_data/data/usstock/`
4. Pass via `--pairs` or add to the config whitelist

New tickers are automatically recognized — no exchange configuration needed.

---

## Repository Layout

```
PortfolioBench/
├── strategy/                  # 8 trading strategies (IStrategy implementations)
├── user_data/strategies/      # 8 portfolio algorithms
├── alpha/                     # 5 pluggable alpha factors (IAlpha interface)
├── portfolio/                 # Standalone portfolio construction pipeline
├── freqtrade/exchange/
│   ├── portfoliobench.py      # Multi-asset exchange (extends Binance)
│   └── polymarket.py          # Polymarket prediction market exchange
├── benchmark.py               # Benchmarking suite with formatted reports
├── benchmark_all.py           # Full benchmark matrix runner
├── tests/                     # Unit and integration tests
├── user_data/data/usstock/    # 357 OHLCV feather files (download from Google Drive)
└── utils/                     # Bash helpers for backtesting and data generation
```

## Tests

```bash
# Run the full test harness (4 asset categories x 3 timeframes = 12 configs)
bash utils/backtest_tests.bash

# Run unit tests directly
python -m pytest tests/ -v
```

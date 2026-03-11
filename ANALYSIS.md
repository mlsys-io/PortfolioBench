# PortfolioBench — Codebase Analysis

## 1. Executive Summary

PortfolioBench is a **multi-asset portfolio benchmarking framework** built as a thin wrapper around [freqtrade](https://github.com/freqtrade/freqtrade). It extends freqtrade's cryptocurrency-focused backtesting engine to support **US equities, global market indices, and mixed-asset portfolios**, while adding pluggable alpha-factor interfaces, academic portfolio-optimization algorithms, and a standalone portfolio-construction pipeline.

### Key Capabilities
- **Multi-asset backtesting**: Crypto, US stocks (~100 tickers), and global indices (DJI, S&P 500, FTSE, Nikkei, etc.)
- **8 portfolio algorithms**: ONS, Minimum Variance, Inverse Volatility, Best Single Asset, Exponential Gradient, Maximum Sharpe, Risk Parity, Polymarket Portfolio
- **8 trading strategies**: EMA Crossover, MACD+ADX, Ichimoku Cloud, RSI+Bollinger, Stochastic+CCI, MLP Speculative, Polymarket Mean Reversion, Polymarket Momentum
- **Alpha factor abstraction**: Decoupled indicator computation via `IAlpha` interface (EmaAlpha, PolymarketAlpha)
- **Blended portfolio construction**: Standalone pipeline combining ONS + EMA + Equal-Weight
- **Automated benchmarking**: Scripts to run strategies across asset classes and timeframes

---

## 2. Architecture Overview

```
PortfolioBench/
├── freqtrade/                  # Vendored from upstream freqtrade (unmodified)
│   └── exchange/
│       ├── portfoliobench.py   # Custom exchange subclass (extends Binance)
│       └── polymarket.py       # Polymarket exchange subclass
│
├── alpha/                      # NEW: Pluggable alpha-factor system
│   ├── interface.py            # IAlpha abstract base class
│   ├── SimpleEmaFactors.py     # EmaAlpha: EMA fast/slow/exit + rolling mean volume
│   └── PolymarketFactors.py    # PolymarketAlpha: prediction-market factors
│
├── strategy/                   # NEW: Freqtrade IStrategy implementations (8 strategies)
│   ├── EmaCrossStrategy.py     # EMA crossover entry/exit strategy
│   ├── MacdAdxStrategy.py      # MACD + ADX trend-confirmation strategy
│   ├── IchimokuCloudStrategy.py        # Ichimoku Cloud strategy
│   ├── RsiBollingerStrategy.py         # RSI + Bollinger Bands strategy
│   ├── StochasticCciStrategy.py        # Stochastic + CCI strategy
│   ├── MlpSpeculativeStrategy.py       # MLP-based speculative strategy
│   ├── mlp_speculative_model/          # MLP model utilities
│   ├── PolymarketMeanReversionStrategy.py  # Polymarket mean reversion
│   └── PolymarketMomentumStrategy.py       # Polymarket momentum
│
├── portfolio/                  # NEW: Standalone portfolio pipeline
│   └── PortfolioManagement.py  # 7-step: load → alpha → signals → ONS → blend → backtest → metrics
│
├── dataset/                    # NEW: Placeholder for data management
│   └── main.py
│
├── tests/                      # Unit and integration tests
│   ├── test_alpha.py           # Alpha factor tests
│   ├── test_data_integrity.py  # Data integrity tests
│   └── test_portfolio_management.py  # Portfolio pipeline tests
│
├── benchmark.py                # Single strategy benchmarking script
├── benchmark_all.py            # Full benchmarking matrix runner
│
├── user_data/
│   ├── config.json             # Backtesting configuration (portfoliobench exchange)
│   ├── config_polymarket.json  # Polymarket backtesting configuration
│   ├── strategies/             # NEW: Portfolio-optimization strategies (8 algorithms)
│   │   ├── ONS.py              # Online Newton Step rebalancing
│   │   ├── inv_vol.py          # Inverse Volatility allocation
│   │   ├── min_var.py          # Minimum Variance allocation
│   │   ├── best_single_asset.py # Momentum rotation (winner-takes-all)
│   │   ├── exp_gradient.py     # Exponential Gradient allocation
│   │   ├── max_sharpe.py       # Maximum Sharpe Ratio optimization
│   │   ├── risk_parity.py      # Risk Parity allocation
│   │   └── polymarket_portfolio.py # Polymarket portfolio strategy
│   └── data/usstock/           # OHLCV data from Google Drive (357 feather files)
│       ├── BTC_USDT-{5m,4h,1d}.feather    # Crypto (10 assets, _USDT suffix)
│       ├── AAPL_USD-{5m,4h,1d}.feather    # US Stocks (~100 assets, _USD suffix)
│       └── DJI_USD-{5m,4h,1d}.feather     # Indices (9 indices, _USD suffix)
│
└── utils/
    ├── backtest_script.bash    # Simple CLI backtest launcher
    ├── backtest_tests.bash     # Comprehensive test harness (4 categories × 3 timeframes)
    ├── backtest_polymarket.bash        # Polymarket backtesting script
    ├── download_polymarket_data.py     # Polymarket data downloader
    ├── generate_polymarket_test_data.py # Polymarket test data generator
    ├── generate_test_data.py           # General test data generator
    └── test.py                         # Test runner
```

---

## 3. Relationship to freqtrade

### 3.1 What Was Copied
The entire `freqtrade/` package is vendored from upstream freqtrade (commit `ed22b4e`, develop branch). This includes:
- Backtesting engine (`freqtrade/optimize/backtesting.py`)
- Strategy interface (`freqtrade/strategy/interface.py`)
- Data handling (`freqtrade/data/`)
- Exchange abstraction (`freqtrade/exchange/`)
- Persistence layer (`freqtrade/persistence/`)
- RPC/API server (`freqtrade/rpc/`)

### 3.2 Custom Exchange Subclass (Clean Extension)
Non-crypto asset support is implemented via **`freqtrade/exchange/portfoliobench.py`**, a clean exchange subclass that extends `Binance`. The vendored `freqtrade/exchange/exchange.py` is **unmodified**.

The `Portfoliobench` subclass handles:

| Feature | Purpose |
|---------|---------|
| **Offline-tolerant market loading** | 5s timeout, 0 retries, graceful fallback for offline use |
| **Synthetic market injection** | Auto-injects stock/index tickers into `self._markets` as valid pairs with permissive precision/limits |
| **Fee calculation fallback** | Returns `0.0` fee for assets without exchange fee data |
| **Leverage tiers fallback** | Returns default 1x leverage tier for non-crypto assets |

To use this exchange, set `"exchange": {"name": "portfoliobench"}` in your config.

### 3.3 What Was NOT Modified
- **Backtesting engine**: Zero changes — all portfolio logic lives in strategy callbacks
- **Data providers**: Zero changes — pre-downloaded feather files match freqtrade's native format
- **Strategy interface**: Zero changes — all new strategies implement `IStrategy` cleanly
- **Configuration system**: Zero changes — uses standard freqtrade JSON config
- **Exchange base class**: Zero changes — `exchange.py` is unmodified; all custom behavior is in the subclass

### 3.4 Design Philosophy
PortfolioBench achieves multi-asset support with **minimal invasiveness**: a clean exchange subclass handles non-crypto tickers, while the vendored freqtrade remains unmodified. All new functionality is added through freqtrade's existing extension points (strategies, callbacks, data format). You can update the vendored freqtrade without re-applying patches.

---

## 4. Alpha Factor System

### 4.1 IAlpha Interface (`alpha/interface.py`)
```python
class IAlpha(ABC):
    def __init__(self, dataframe: DataFrame, metadata: dict = {}):
        self.dataframe = dataframe
        self.metadata = metadata

    @abstractmethod
    def process(self) -> DataFrame:
        """Decouples indicator computation from IStrategy"""
        pass
```
**Design pattern**: Strategy pattern — allows swappable indicator computation without modifying strategy code.

### 4.2 EmaAlpha (`alpha/SimpleEmaFactors.py`)
Concrete implementation computing:
- `ema_fast` (period 12, optimizable 5-15)
- `ema_slow` (period 26, optimizable 20-30)
- `ema_exit` (period 6, optimizable 5-10)
- `mean-volume` (20-period rolling mean)

Uses freqtrade's `IntParameter` for hyperparameter optimization compatibility.

---

## 5. Trading Strategies

### 5.1 EmaCrossStrategy (`strategy/EmaCrossStrategy.py`)
| Attribute | Value |
|-----------|-------|
| Type | Trend-following |
| Timeframe | Configurable (default from config) |
| Entry | EMA fast crosses above EMA slow AND volume > 0.75× mean |
| Exit | EMA exit crosses below EMA fast |
| Stop loss | -99 (effectively none) |
| Reference | http://arxiv.org/abs/2511.00665 |

Delegates indicator computation to `EmaAlpha`, demonstrating the alpha factor abstraction.

### 5.2 MacdAdxStrategy (`strategy/MacdAdxStrategy.py`)
| Attribute | Value |
|-----------|-------|
| Type | Trend-confirmation |
| Timeframe | 5m |
| Entry | MACD > signal AND ADX > 25 AND volume filter |
| Exit | MACD < signal AND ADX > 25 AND volume filter |
| Stop loss | -99 (effectively none) |
| Parameters | macdFast(8-15), macdSlow(20-30), macdSignal(10-15), adxPeriod(10-20) |

### 5.3 Additional Trading Strategies

| Strategy | Type | Key Indicators |
|----------|------|---------------|
| **IchimokuCloudStrategy** | Trend-following | Ichimoku Cloud (Tenkan, Kijun, Senkou spans) |
| **RsiBollingerStrategy** | Mean-reversion | RSI + Bollinger Bands |
| **StochasticCciStrategy** | Oscillator-based | Stochastic Oscillator + CCI |
| **MlpSpeculativeStrategy** | ML-based | MLP neural network predictions |
| **PolymarketMeanReversionStrategy** | Mean-reversion | Polymarket prediction-market signals |
| **PolymarketMomentumStrategy** | Momentum | Polymarket prediction-market momentum |

Multiple strategies include `confirm_trade_entry()` with 1% price deviation guard.

---

## 6. Portfolio Algorithms

### 6.1 ONS — Online Newton Step (`user_data/strategies/ONS.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Online convex optimization (Newton Step) |
| Rebalance | Every candle (continuous) |
| Parameters | eta=0.0, beta=1.0, delta=0.125 |
| Constraint | Weights sum to 0.95 (5% cash reserve) |
| Integration | `custom_stake_amount()` + `adjust_trade_position()` |

**How it works**: Maintains a running Hessian matrix `A` and gradient accumulator `b`. Each period, computes optimal weights by solving a constrained quadratic program (projection onto the probability simplex under the A-norm). Adapts to changing market conditions by adjusting weights based on realized returns.

### 6.2 Inverse Volatility (`user_data/strategies/inv_vol.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Weight ∝ 1/σ (inverse rolling volatility) |
| Rebalance | Monthly (1st trading day) |
| Lookback | 30 days |
| Timeframe | 1d |

**How it works**: Over a 30-day window, computes each asset's return standard deviation, then allocates inversely proportional to volatility. Low-volatility assets receive higher weights.

### 6.3 Minimum Variance (`user_data/strategies/min_var.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | w = Σ⁻¹1 / (1ᵀΣ⁻¹1) — global minimum variance |
| Rebalance | Monthly (1st trading day) |
| Lookback | 30 days |
| Timeframe | 1d |

**How it works**: Estimates the covariance matrix from rolling returns, then solves for the portfolio that minimizes total variance. Uses pseudo-inverse (`np.linalg.pinv`) to handle singular/near-singular covariance matrices. Negative weights are clipped to zero.

### 6.4 Best Single Asset (`user_data/strategies/best_single_asset.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Momentum rotation — hold the single best-performing asset |
| Rebalance | Monthly (day 1) |
| Lookback | 90 days |
| Max positions | 1 |

**How it works**: Computes 90-day momentum (price return) for all whitelist pairs via `informative_pairs()`. On rebalance day, enters the pair with highest momentum and exits the current holding if it's no longer the best.

### 6.5 Exponential Gradient (`user_data/strategies/exp_gradient.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Multiplicative weight update (exponential gradient) |
| Rebalance | Every candle (continuous) |
| Integration | `custom_stake_amount()` + `adjust_trade_position()` |

### 6.6 Maximum Sharpe (`user_data/strategies/max_sharpe.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Maximum Sharpe ratio optimization |
| Rebalance | Monthly (1st trading day) |
| Lookback | 30 days |

### 6.7 Risk Parity (`user_data/strategies/risk_parity.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Equal risk contribution across assets |
| Rebalance | Monthly (1st trading day) |
| Lookback | 30 days |

### 6.8 Polymarket Portfolio (`user_data/strategies/polymarket_portfolio.py`)
| Attribute | Value |
|-----------|-------|
| Algorithm | Prediction-market weighted allocation |
| Rebalance | Monthly |

### 6.9 Portfolio Strategy Comparison

| Strategy | Allocation | Rebalance | Positions | Optimization | Risk Model |
|----------|------------|-----------|-----------|--------------|------------|
| ONS | Convex optimization | Per-candle | All pairs | Online learning | Adaptive Hessian |
| Exp Gradient | Multiplicative update | Per-candle | All pairs | Online learning | Return-based |
| Inverse Vol | 1/volatility | Monthly | All pairs | None (analytical) | Rolling σ |
| Min Variance | Inv-covariance | Monthly | All pairs | Quadratic | Rolling Σ |
| Max Sharpe | Sharpe maximization | Monthly | All pairs | Mean-variance | Rolling μ, Σ |
| Risk Parity | Equal risk contribution | Monthly | All pairs | Risk budgeting | Rolling Σ |
| Best Single | Winner-takes-all | Monthly | 1 pair | None (ranking) | Momentum |
| Polymarket | Prediction-market weighted | Monthly | All pairs | Market-implied | Prediction odds |
| EMA Cross | Signal-based | On signal | Per-pair | None | Trend |
| MACD+ADX | Signal-based | On signal | Per-pair | None | Trend + ADX |
| Ichimoku Cloud | Signal-based | On signal | Per-pair | None | Cloud trend |
| RSI+Bollinger | Signal-based | On signal | Per-pair | None | Mean-reversion |
| Stochastic+CCI | Signal-based | On signal | Per-pair | None | Oscillator |
| MLP Speculative | Signal-based | On signal | Per-pair | ML model | Neural network |
| Polymarket MeanRev | Signal-based | On signal | Per-pair | None | Prediction-market |
| Polymarket Momentum | Signal-based | On signal | Per-pair | None | Prediction-market |

---

## 7. Standalone Portfolio Pipeline (`portfolio/PortfolioManagement.py`)

A self-contained 7-step pipeline that operates **outside** freqtrade's backtesting engine:

```
Step 1: Load OHLCV feather data
Step 2: Generate EMA alpha indicators (via EmaAlpha)
Step 3: Compute EMA cross entry/exit signals → binary positions
Step 4: Compute ONS weights (Online Newton Step)
Step 5: Set up 1/N equal-weight allocation
Step 6: Blend strategies (34% equal + 33% ONS + 33% EMA)
Step 7: Walk-forward backtest → Sharpe, max drawdown, returns
```

**Blending formula** (per bar, per asset):
```
final_w[pair] = 0.34 × (1/N) + 0.33 × ons_weight[pair] + 0.33 × ema_position[pair] × (1/N)
```
Weights are re-normalized to sum to 1 each bar.

**Metrics computed**: Total return, annualized return, annualized Sharpe ratio, max drawdown.

---

## 8. Multi-Asset Data Infrastructure

### 8.1 Asset Universe (119 instruments, 357 feather files)

**Cryptocurrencies (10)**:
BTC, ETH, SOL, XRP, DOGE, BNB, ADA, TRX, STETH, BCH

**US Stocks (~100, roughly S&P 100)**:
AAPL, MSFT, NVDA, GOOG, AMZN, META, TSLA, JPM, MA, V, UNH, HD, PG, JNJ, LLY, AVGO, COST, NFLX, ORCL, CRM, AMD, INTC, MU, QCOM, TXN, ...

**Global Indices (9)**:
DJI (Dow Jones), GSPC (S&P 500), IXIC (Nasdaq), RUT (Russell 2000), FTSE (UK), N225 (Nikkei), HSI (Hang Seng), STOXX50E (Euro Stoxx 50), VIX

### 8.2 Data Format
All data stored as feather files in `user_data/data/usstock/`:
- Crypto: `{TICKER}_USDT-{timeframe}.feather` (e.g. `BTC_USDT-1d.feather`)
- Stocks & indices: `{TICKER}_USD-{timeframe}.feather` (e.g. `AAPL_USD-1d.feather`)

Timeframes: `5m`, `4h`, `1d`

### 8.3 How Stock/Index Data Works in freqtrade
The `Portfoliobench` exchange subclass auto-injects any pair from the whitelist that isn't a real Binance pair into `self._markets` with synthetic market metadata. This means:
- freqtrade's pair validation passes
- Data loading works (just needs matching filenames)
- Fee calculation returns 0.0 for non-exchange pairs
- Leverage lookups return default 1x

### 8.4 Test Harness (`utils/backtest_tests.bash`)
Defines 4 asset categories × 3 timeframes = 12 test configurations:
- `crypto_only`: BTC, ETH, SOL, XRP
- `stock_only`: AAPL, MSFT, NVDA, GOOG
- `index_only`: DJI, FTSE, GSPC
- `mix_assets`: All of the above combined

---

## 9. Backtesting Skills & Strategies

### 9.1 Running a Basic Backtest

```bash
# Activate environment
source .venv/bin/activate

# Crypto-only backtest with EMA Cross strategy
portbench backtesting \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --timeframe 4h \
    --timerange 20250501-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT XRP/USDT

# Stock backtest with MACD+ADX strategy
portbench backtesting \
    --strategy MacdAdxStrategy \
    --strategy-path ./strategy \
    --timeframe 1d \
    --timerange 20240101-20260131 \
    --pairs AAPL/USD MSFT/USD NVDA/USD GOOG/USD

# Portfolio strategy (ONS) across mixed assets
portbench backtesting \
    --strategy ONS_Portfolio \
    --strategy-path ./user_data/strategies \
    --timeframe 5m \
    --timerange 20260101-20260108 \
    --pairs BTC/USDT ETH/USDT AAPL/USD MSFT/USD DJI/USD \
    --dry-run-wallet 1000000
```

### 9.2 Running the Standalone Portfolio Pipeline

```bash
cd /path/to/PortfolioBench
python -m portfolio.PortfolioManagement
```

This runs the full 7-step pipeline with default parameters:
- Pairs: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT, MSFT/USDT
- Timeframe: 1d
- Initial capital: $10,000
- Blend: 34% equal-weight + 33% ONS + 33% EMA

### 9.3 Running the Comprehensive Test Suite

```bash
# Run all 12 test configurations (4 categories × 3 timeframes)
bash utils/backtest_tests.bash

# Or use the simple script with custom parameters
bash utils/backtest_script.bash -s EmaCrossStrategy -a "BTC/USDT ETH/USDT"
```

### 9.4 Hyperparameter Optimization

```bash
# Optimize EMA parameters
portbench hyperopt \
    --strategy EmaCrossStrategy \
    --strategy-path ./strategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --timeframe 4h \
    --timerange 20250101-20250601 \
    --pairs BTC/USDT ETH/USDT SOL/USDT \
    --epochs 100

# Optimize MACD+ADX parameters
portbench hyperopt \
    --strategy MacdAdxStrategy \
    --strategy-path ./strategy \
    --hyperopt-loss SharpeHyperOptLoss \
    --timeframe 5m \
    --timerange 20260101-20260108 \
    --pairs AAPL/USD MSFT/USD NVDA/USD \
    --epochs 100
```

---

## 10. Model/Strategy/Portfolio Benchmarking

### 10.1 Strategy Benchmarking Dimensions

| Dimension | Values |
|-----------|--------|
| **Strategies** | EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative, PolymarketMeanReversion, PolymarketMomentum, ONS, ExpGradient, InverseVol, MinVar, MaxSharpe, RiskParity, BestSingleAsset, PolymarketPortfolio |
| **Asset classes** | Crypto, US Stocks, Global Indices, Mixed |
| **Timeframes** | 5m, 4h, 1d |
| **Universes** | Crypto-only (10), Stock-only (~100), Index-only (9), All (119) |
| **Time periods** | Various (2024-2026 data available) |

### 10.2 Benchmark Metrics
- **Total return** (%)
- **Annualized return** (%)
- **Annualized Sharpe ratio**
- **Maximum drawdown** (%)
- **Number of trades**
- **Win rate**
- **Profit factor**

### 10.3 Benchmarking Matrix

To fully benchmark, run each strategy against each asset category and timeframe:

```
For strategy in [EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative,
                  PolymarketMeanReversion, PolymarketMomentum,
                  ONS, ExpGradient, InverseVol, MinVar, MaxSharpe, RiskParity, BestSingleAsset,
                  PolymarketPortfolio]:
  For category in [crypto_only, stock_only, index_only, mix_assets]:
    For timeframe in [5m, 4h, 1d]:
      Run backtest → collect metrics
```

This produces 16 × 4 × 3 = **192 benchmark configurations**.

### 10.4 Adding New Strategies

1. **Create a new alpha factor** (optional):
```python
# alpha/MyAlpha.py
from alpha.interface import IAlpha

class MyAlpha(IAlpha):
    def process(self):
        self.dataframe["my_indicator"] = ...
        return self.dataframe
```

2. **Create a new trading strategy**:
```python
# strategy/MyStrategy.py
from freqtrade.strategy import IStrategy

class MyStrategy(IStrategy):
    INTERFACE_VERSION = 3

    def populate_indicators(self, dataframe, metadata):
        # Use alpha factors or compute directly
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[condition, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe.loc[condition, "exit_long"] = 1
        return dataframe
```

3. **Create a new portfolio algorithm**:
```python
# user_data/strategies/MyPortfolio.py
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade

class MyPortfolio(IStrategy):
    INTERFACE_VERSION = 3
    position_adjustment_enable = True

    def populate_indicators(self, dataframe, metadata):
        # Compute target_weight for this pair
        dataframe['target_weight'] = ...
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe.loc[:, 'enter_long'] = 1  # Always enter
        return dataframe

    def adjust_trade_position(self, trade, current_time, current_rate, ...):
        # Rebalance to target weight
        target_size = total_wallet * target_weight
        diff = target_size - current_position_value
        return diff if abs(diff) > threshold else None
```

### 10.5 Adding New Asset Classes

1. Prepare OHLCV data as feather files with columns: `date, open, high, low, close, volume`
2. Name files as `{TICKER}_USDT-{timeframe}.feather` (crypto) or `{TICKER}_USD-{timeframe}.feather` (stocks/indices)
3. Place in `user_data/data/usstock/`
4. Add pairs to config whitelist or pass via `--pairs` CLI flag
5. The `Portfoliobench` exchange subclass will auto-inject synthetic market entries

---

## 11. Key Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Strategy (GoF)** | `IAlpha` → `EmaAlpha` | Pluggable indicator computation |
| **Template Method** | `IStrategy` lifecycle | freqtrade dictates: indicators → entry → exit |
| **Adapter** | `portfoliobench.py` subclass | Makes stock data compatible with crypto infrastructure |
| **Caching** | All portfolio strategies | Avoids redundant weight computation across pairs |
| **Pipeline** | `PortfolioManagement.py` | Sequential data → indicators → signals → weights → backtest |
| **Facade** | `run_portfolio()` | Single entry point for the complete pipeline |

---

## 12. Limitations and Future Work

### Current Limitations
1. **No short selling**: All strategies are long-only (`can_short = False`)
2. **No transaction costs for stocks**: Fee = 0.0 (unrealistic for real trading)
3. **Single exchange format**: All data must use Binance naming convention
4. **No live trading**: Designed for backtesting only (exchange subclass is for offline use)
5. **Dataset module is a stub**: `dataset/main.py` is a placeholder

### Future Opportunities
1. **ML Strategy Integration**: Connect FreqAI or custom ML models via the `IAlpha` interface
2. **Realistic costs**: Model bid-ask spreads, market impact, and brokerage commissions
3. **Risk management**: Add drawdown limits, position size limits, sector exposure limits
4. **Walk-forward optimization**: Time-series cross-validation for parameter tuning
5. **Multi-timeframe strategies**: Combine signals across 5m/4h/1d
6. **Automated benchmarking**: Script that runs all 192 configurations and produces comparison tables

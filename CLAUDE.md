# PortfolioBench — Developer Guide

## Project Overview
PortfolioBench is a multi-asset portfolio benchmarking framework built on top of freqtrade (included as a git submodule). It extends freqtrade to support US stocks, global indices, prediction markets, and portfolio optimization algorithms beyond cryptocurrency trading.

## Setup
```bash
git clone --recurse-submodules https://github.com/mlsys-io/PortfolioBench.git
cd PortfolioBench
pip install -e .
```

Or if already cloned:
```bash
git submodule update --init --recursive
pip install -e .
```

## Repository Layout
- `freqtrade/` — Git submodule → [mlsys-io/freqtrade](https://github.com/mlsys-io/freqtrade) (PortfolioBench-specific changes: `exchange/portfoliobench.py`, `exchange/polymarket.py`, CLI subcommands)
- `alpha/` — Pluggable alpha-factor interface (`IAlpha`) and implementations (EmaAlpha, RsiAlpha, MacdAlpha, BollingerAlpha, PolymarketAlpha)
- `strategy/` — Freqtrade `IStrategy` implementations (EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative, Polymarket strategies)
- `portfolio/` — Standalone portfolio construction pipeline
- `tests/` — Unit and integration tests (alpha, data integrity, portfolio management)
- `benchmark.py` / `benchmark_all.py` — Benchmarking scripts (also accessible via `portbench benchmark`)
- `user_data/strategies/` — Portfolio-optimization strategies (ONS, MinVar, InvVol, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity, Polymarket)
- `user_data/config.json` — Main backtesting config; `user_data/config_polymarket.json` — Polymarket config
- `user_data/data/usstock/` — OHLCV feather files (119 instruments x 3 timeframes = 357 files; download from Google Drive)
- `user_data/data/polymarket/` — Polymarket event contract feather files (generated via `portbench generate-data` or downloaded)
- `utils/` — Bash scripts for backtesting, data generation, and testing

## Key Commands

```bash
# Backtest a trading strategy
portbench backtesting --strategy EmaCrossStrategy --strategy-path ./strategy --timeframe 4h --timerange 20250101-20250601 --pairs BTC/USDT ETH/USDT

# Backtest a portfolio strategy
portbench backtesting --strategy ONS_Portfolio --strategy-path ./user_data/strategies --timeframe 5m --timerange 20260101-20260108 --pairs BTC/USDT ETH/USDT AAPL/USD --dry-run-wallet 1000000

# Run standalone portfolio pipeline
portbench portfolio

# Generate synthetic test data (all asset classes including Polymarket)
portbench generate-data

# Run unit tests
python -m pytest tests/ -v

# Run full backtest test suite
bash utils/backtest_tests.bash
```

## Existing Strategies
- **Trading** (in `strategy/`): EmaCross, MacdAdx, IchimokuCloud, RsiBollinger, StochasticCci, MlpSpeculative, PolymarketMeanReversion, PolymarketMomentum
- **Portfolio** (in `user_data/strategies/`): ONS, InverseVol, MinVar, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity, PolymarketPortfolio

## Adding New Strategies
1. For alpha factors: implement `IAlpha.process()` in `alpha/`
2. For trading strategies: implement `IStrategy` in `strategy/`
3. For portfolio algorithms: implement `IStrategy` with `position_adjustment_enable=True` in `user_data/strategies/`

## Adding New Assets
Place feather files in `user_data/data/usstock/`:
- Crypto: `{TICKER}_USDT-{timeframe}.feather` (e.g. `BTC_USDT-1d.feather`)
- Stocks & indices: `{TICKER}_USD-{timeframe}.feather` (e.g. `AAPL_USD-1d.feather`)

The `Portfoliobench` exchange subclass auto-injects synthetic market entries for any pair not found on the real exchange.

## Custom Exchange: `Portfoliobench`
Non-crypto asset support is implemented via a clean exchange subclass at `freqtrade/freqtrade/exchange/portfoliobench.py` (extends `Binance`). It handles:
- Offline-tolerant market loading (5s timeout, 0 retries, graceful fallback)
- Synthetic market injection for stocks/indices (any pair in the whitelist or CLI)
- Proper quote-currency convention: crypto uses USDT, stocks/indices use USD
- USD/USDT normalisation so both work with a single `stake_currency` setting
- Zero-fee fallback for assets without exchange fee data
- Default 1x leverage tier for non-crypto assets

To use this exchange, set `"exchange": {"name": "portfoliobench"}` in your config.

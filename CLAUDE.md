# PortfolioBench — Developer Guide

## Project Overview
PortfolioBench is a multi-asset portfolio benchmarking framework wrapping freqtrade. It extends freqtrade to support US stocks, global indices, and portfolio optimization algorithms beyond cryptocurrency trading.

## Repository Layout
- `freqtrade/` — Vendored freqtrade (unmodified; custom behaviour lives in `exchange/portfoliobench.py`)
- `alpha/` — Pluggable alpha-factor interface (`IAlpha`) and implementations (EmaAlpha, PolymarketFactors)
- `strategy/` — Freqtrade `IStrategy` implementations (EmaCross, MacdAdx, Ichimoku, RsiBollinger, StochasticCci, MlpSpeculative, Polymarket strategies)
- `portfolio/` — Standalone portfolio construction pipeline
- `dataset/` — Data management (stub)
- `user_data/strategies/` — Portfolio-optimization strategies (ONS, MinVar, InvVol, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity, Polymarket)
- `user_data/data/binance/` — Pre-downloaded OHLCV feather files (119 instruments × 3 timeframes)
- `utils/` — Bash scripts for backtesting, data generation, and testing

## Key Commands

```bash
# Backtest a trading strategy
freqtrade backtesting --strategy EmaCrossStrategy --strategy-path ./strategy --timeframe 4h --timerange 20250101-20250601 --pairs BTC/USDT ETH/USDT

# Backtest a portfolio strategy
freqtrade backtesting --strategy ONS_Portfolio --strategy-path ./user_data/strategies --timeframe 5m --timerange 20260101-20260108 --pairs BTC/USDT ETH/USDT AAPL/USDT --dry-run-wallet 1000000

# Run standalone portfolio pipeline
python -m portfolio.PortfolioManagement

# Run full test suite
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
Place feather files as `{TICKER}_USDT-{timeframe}.feather` in `user_data/data/binance/`. The `Portfoliobench` exchange subclass auto-injects synthetic market entries for any pair not found on the real exchange.

## Custom Exchange: `Portfoliobench`
Non-crypto asset support is implemented via a clean exchange subclass at `freqtrade/exchange/portfoliobench.py` (extends `Binance`). It handles:
- Offline-tolerant market loading (5s timeout, 0 retries, graceful fallback)
- Synthetic market injection for stocks/indices (any pair in the whitelist or CLI)
- Zero-fee fallback for assets without exchange fee data
- Default 1x leverage tier for non-crypto assets

The vendored `freqtrade/exchange/exchange.py` is **unmodified** — you can update the vendored freqtrade without re-applying patches. To use this exchange, set `"exchange": {"name": "portfoliobench"}` in your config.

# PortfolioBench — Developer Guide

## Project Overview
PortfolioBench is a multi-asset portfolio benchmarking framework wrapping freqtrade. It extends freqtrade to support US stocks, global indices, and portfolio optimization algorithms beyond cryptocurrency trading.

## Repository Layout
- `freqtrade/` — Vendored freqtrade (only `exchange/exchange.py` is modified)
- `alpha/` — Pluggable alpha-factor interface (`IAlpha`) and implementations
- `strategy/` — Freqtrade `IStrategy` implementations (EMA Cross, MACD+ADX)
- `portfolio/` — Standalone portfolio construction pipeline
- `user_data/strategies/` — Portfolio-optimization strategies (ONS, MinVar, InvVol, BestSingleAsset)
- `user_data/data/binance/` — Pre-downloaded OHLCV feather files (119 instruments × 3 timeframes)
- `utils/` — Bash scripts for backtesting

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

## Adding New Strategies
1. For alpha factors: implement `IAlpha.process()` in `alpha/`
2. For trading strategies: implement `IStrategy` in `strategy/`
3. For portfolio algorithms: implement `IStrategy` with `position_adjustment_enable=True` in `user_data/strategies/`

## Adding New Assets
Place feather files as `{TICKER}_USDT-{timeframe}.feather` in `user_data/data/binance/`. The exchange.py hack auto-injects synthetic market entries for non-crypto tickers.

## Important: exchange.py Hacks
`freqtrade/exchange/exchange.py` has 4 hacks (marked with `# HACK`) to support non-crypto assets. Do NOT update the vendored freqtrade without re-applying these patches.

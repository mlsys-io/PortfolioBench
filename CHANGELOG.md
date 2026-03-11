# Changelog

## v0.1.0 — 2026-03-11

Initial release of PortfolioBench, a multi-asset portfolio benchmarking framework built on top of freqtrade.

### Highlights

- **Multi-asset support** — backtest across cryptocurrencies, US equities (119 instruments), global indices, and Polymarket prediction markets in a single framework
- **16 built-in strategies** — 8 trading strategies and 8 portfolio allocation algorithms ready to use out of the box
- **5 alpha factors** — pluggable alpha-factor interface with EMA, RSI, MACD, Bollinger Bands, and Polymarket implementations
- **Full benchmarking suite** — run all strategies across multiple asset classes and timeframes with one command, with parallel execution support

### Features

- **Custom exchange (`Portfoliobench`)** — clean exchange subclass extending Binance with offline-tolerant market loading, synthetic market injection for stocks/indices, and USD/USDT normalization
- **Trading strategies**: EmaCross, MacdAdx, IchimokuCloud, RsiBollinger, StochasticCci, MlpSpeculative, PolymarketMeanReversion, PolymarketMomentum
- **Portfolio strategies**: ONS (Online Newton Step), InverseVol, MinVar, BestSingleAsset, ExpGradient, MaxSharpe, RiskParity, PolymarketPortfolio
- **Alpha factors**: EmaAlpha, RsiAlpha, MacdAlpha, BollingerAlpha, PolymarketAlpha with `IAlpha` interface
- **`portbench` CLI** — unified command-line interface with `backtesting`, `portfolio`, `benchmark`, and `generate-data` subcommands
- **Benchmark report** — HTML report generation with detailed metrics per strategy
- **Parallel backtest execution** — `--workers` flag for concurrent benchmark runs
- **Google Drive data integration** — automatic download of OHLCV data (119 instruments × 3 timeframes)
- **GitHub Actions CI** — unit tests, strategy import validation, benchmark report generation, and GitHub Pages deployment
- **Polymarket integration** — backtest strategies on binary outcome prediction market contracts
- **Standalone portfolio pipeline** — run portfolio construction outside of freqtrade's backtest loop
- **Synthetic data generation** — `portbench generate-data` for testing without real market data

# About PortfolioBench

**The open-source framework for backtesting portfolios across every asset class — from Bitcoin to blue chips to prediction markets.**

PortfolioBench lets you evaluate trading strategies and portfolio allocation algorithms on a unified, multi-asset dataset of 119 instruments spanning crypto, US equities, global indices, and Polymarket event contracts. One command, one framework, one consistent evaluation — no more stitching together fragmented tools.

## Why PortfolioBench?

- **Multi-asset from day one.** Test a single portfolio that holds BTC, AAPL, the Nikkei 225, and a "Will ETH hit $10K?" prediction contract — all in one backtest.
- **16 strategies, ready to run.** Eight trading strategies (EMA crossover, MACD+ADX, Ichimoku, ML-based, and more) plus eight portfolio algorithms (Online Newton Step, Risk Parity, Max Sharpe, Exponential Gradient, and others).
- **192 benchmark configurations.** 16 strategies × 4 asset classes × 3 timeframes — compare everything head-to-head with a single command.
- **Prediction markets included.** First-class support for Polymarket binary outcome contracts, opening an entirely new asset class for systematic backtesting.
- **Built on battle-tested infrastructure.** Extends [freqtrade](https://github.com/freqtrade/freqtrade), one of the most widely used open-source trading engines, so you inherit its reliability, community, and ecosystem.

## Who is it for?

- **Quantitative researchers** evaluating portfolio construction methods across asset classes.
- **Algorithmic traders** prototyping and stress-testing strategies before going live.
- **Students and academics** studying portfolio theory with real, multi-asset data.
- **Prediction market enthusiasts** applying systematic trading to event contracts.
- **Anyone** who believes diversification should extend beyond a single asset class.

## Get started in 60 seconds

```bash
git clone --recurse-submodules https://github.com/mlsys-io/PortfolioBench.git
cd PortfolioBench
pip install -e .
portbench benchmark --quick
```

That's it — you're running backtests across crypto, stocks, indices, and prediction markets.

## License & Contributing

PortfolioBench is open source. Contributions are welcome — whether it's a new strategy, a new asset class, better documentation, or a bug fix. See the [README](README.md) to get oriented, then open a PR.

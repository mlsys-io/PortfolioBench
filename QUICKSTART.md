"""
QUICKSTART.md - Pipeline System Quick Start
============================================

Get up and running with PortfolioBench pipelines in 5 minutes.
"""

# Quick Start: PortfolioBench Pipeline System

## Installation

```bash
# Clone and install
git clone --recurse-submodules https://github.com/mlsys-io/PortfolioBench.git
cd PortfolioBench
pip install -e .

# Download data (optional - can use synthetic data for testing)
pip install gdown
portbench download-data --exchange portfoliobench
```

## 1. Run a Preset Pipeline (Fastest)

```python
from pipeline.integrations import PresetPipelineRunner

# Run simple EMA cross strategy
result = PresetPipelineRunner.run_simple_ema_cross(
    pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    output_dir="./output/ema_cross"
)

# Print results
result.print_summary()
```

## 2. Run from Configuration File

```python
from pipeline.orchestrator import run_pipeline

# JSON config
result = run_pipeline("pipelines/simple_ema_cross.json")

# YAML config
result = run_pipeline("pipelines/my_config.yaml")
```

## 3. Create Custom Configuration

### Python
```python
from pipeline.config import (
    PipelineConfig, AlphaConfig, StrategyConfig,
    PortfolioConfig, BacktestConfig, AlphaType,
    StrategyType, PortfolioAlgorithm
)

config = PipelineConfig(
    name="My Strategy",
    alpha=[AlphaConfig(type=AlphaType.EMA)],
    strategies=[StrategyConfig(type=StrategyType.EMA_CROSS)],
    portfolio=PortfolioConfig(algorithm=PortfolioAlgorithm.EQUAL_WEIGHT),
    backtest=BacktestConfig(
        timerange="20250101-20250601",
        pairs=["BTC/USDT", "ETH/USDT"],
        initial_capital=50000.0
    )
)

result = run_pipeline(config)
```

### JSON
```json
{
  "name": "My Strategy",
  "alpha": [{"type": "ema"}],
  "strategies": [{"type": "ema_cross"}],
  "portfolio": {"algorithm": "equal_weight"},
  "backtest": {
    "timerange": "20250101-20250601",
    "pairs": ["BTC/USDT", "ETH/USDT"],
    "initial_capital": 50000
  }
}
```

### YAML
```yaml
name: My Strategy
alpha:
  - type: ema
strategies:
  - type: ema_cross
portfolio:
  algorithm: equal_weight
backtest:
  timerange: "20250101-20250601"
  pairs: [BTC/USDT, ETH/USDT]
  initial_capital: 50000
```

## 4. Access Results

```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline(config)

# Performance metrics
print(f"Return: {result.metrics['total_return_pct']}%")
print(f"Sharpe: {result.metrics['annualised_sharpe']:.4f}")
print(f"Max Drawdown: {result.metrics['max_drawdown_pct']}%")

# Portfolio weights
weights = result.portfolio_weights  # pd.DataFrame
weights.to_csv("portfolio_weights.csv")

# Backtest results
backtest = result.backtest_result  # pd.DataFrame
backtest.to_csv("backtest_results.csv")

# Save all outputs
result.save_all("./output/my_results")
```

## 5. Compare Multiple Strategies

```python
from pipeline.integrations import (
    BatchPipelineRunner,
    PipelineComparator
)

runner = BatchPipelineRunner()

results = runner.run_multiple([
    "pipelines/simple_ema_cross.json",
    "pipelines/balanced_multi_alpha.json",
    "pipelines/risk_parity.json"
])

# Compare
PipelineComparator.print_comparison(results)
```

## 6. Export to Freqtrade

```python
from pipeline.integrations import FreqtradeIntegration

result = run_pipeline(config)

# Get freqtrade config
ft_config = FreqtradeIntegration.config_from_pipeline(result.config)

# Export results
FreqtradeIntegration.export_backtest_results_for_freqtrade(
    result.backtest_result,
    result.portfolio_weights,
    "./output/freqtrade"
)
```

## Available Preset Configurations

### 1. Simple EMA Cross
```python
from pipeline.config import PresetConfigs
config = PresetConfigs.simple_ema_cross()
```
- Basic EMA crossover on crypto (BTC, ETH, SOL, XRP)
- 4-hour timeframe, 10k initial capital

### 2. Balanced Multi-Alpha
```python
config = PresetConfigs.balanced_multi_alpha()
```
- EMA, RSI, MACD factors
- Mixed assets (crypto + stocks)
- ONS portfolio optimization
- 50k initial capital

### 3. Risk Parity
```python
config = PresetConfigs.risk_parity_portfolio()
```
- Equal risk contribution
- Multi-asset (crypto, equities, bonds, commodities)
- Monthly rebalancing
- 100k initial capital

## Understanding Output

### Metrics
- **Total Return %**: Total portfolio return
- **Annualized Return %**: Compound annual growth rate
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown %**: Worst peak-to-trough decline

### Validation
Pipeline validates:
- ✓ Data integrity (completeness, alignment)
- ✓ Alpha signals (columns, ranges)
- ✓ Strategy signals (binary values)
- ✓ Portfolio weights (sum to 1.0)
- ✓ Backtest results (validity)

View validation results:
```python
print(f"Pass rate: {result.validation['pass_rate']}%")
for check in result.validation['results']:
    print(f"  {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
```

### File Outputs
```
output/
├── pipeline_result.json         # Complete result JSON
├── portfolio_weights.csv        # Weight matrix
├── backtest_results.csv         # Daily portfolio values
└── report.html                  # Interactive report
```

## Common Tasks

### Task 1: Backtest Different Timeframes
```python
config1 = PresetConfigs.simple_ema_cross()
config1.backtest.timeframe = "5m"

config2 = PresetConfigs.simple_ema_cross()
config2.backtest.timeframe = "4h"

config3 = PresetConfigs.simple_ema_cross()
config3.backtest.timeframe = "1d"

from pipeline.integrations import BatchPipelineRunner
runner = BatchPipelineRunner()
results = runner.run_multiple([config1, config2, config3])
```

### Task 2: Test Different Asset Classes
```python
config = PresetConfigs.balanced_multi_alpha()

# Crypto only
config.backtest.pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

# Stocks only
config.backtest.pairs = ["AAPL/USD", "MSFT/USD", "NVDA/USD"]

# Mixed
config.backtest.pairs = ["BTC/USDT", "AAPL/USD", "GLD/USD"]
```

### Task 3: Optimize Capital Allocation
```python
config = PresetConfigs.balanced_multi_alpha()
config.backtest.initial_capital = 100000.0  # Change capital
config.portfolio.strategy_weights = {
    "ema_cross": 0.5,
    "rsi_bollinger": 0.3,
    "macd_adx": 0.2
}
```

### Task 4: Enable Detailed Logging
```python
config = PresetConfigs.simple_ema_cross()
config.verbose = True

result = run_pipeline(config)
```

## Troubleshooting

### Error: No data loaded
```python
# Check data directory
config.data.data_dir = "./user_data/data/usstock"

# Verify files exist: {TICKER}_{QUOTE}-{timeframe}.feather
# Example: BTC_USDT-1d.feather, AAPL_USD-1d.feather
```

### Error: Module not found
```bash
# Install requirements
pip install -e .
pip install -r requirements-ml.txt
```

### Validation failures
```python
# Enable verbose output to see detailed checks
config.verbose = True
result = run_pipeline(config)

# Check validation results
print(result.validation)
```

## Next Steps

1. **Explore Examples**: See `examples/pipeline_examples.py`
2. **Read Documentation**: See `PIPELINE_README.md`
3. **Run Tests**: `pytest tests/test_pipeline.py -v`
4. **Customize**: Modify configurations in `pipelines/`
5. **Integrate**: Export to freqtrade with integrations module

## Getting Help

- Check example configs in `pipelines/`
- Read docstrings: `help(PipelineConfig)`
- View tests: `tests/test_pipeline.py`
- See full documentation: `PIPELINE_README.md`

# PortfolioBench Pipeline System

A comprehensive end-to-end portfolio construction and backtesting pipeline for PortfolioBench.

## Overview

The pipeline system provides:

- **End-to-end orchestration** of alpha → strategy → portfolio → backtest
- **Flexible configuration** via JSON, YAML, or Python
- **Comprehensive validation** at each stage
- **Result aggregation** and reporting (HTML, CSV, JSON)
- **Freqtrade integration** for dashboard and UI
- **Batch execution** for comparing multiple strategies
- **Preset configurations** for common scenarios

## Quick Start

### 1. Run a Preset Pipeline

```python
from pipeline.config import PresetConfigs
from pipeline.orchestrator import run_pipeline

# Simple EMA cross strategy
config = PresetConfigs.simple_ema_cross()
result = run_pipeline(config)

# Balanced multi-alpha strategy
config = PresetConfigs.balanced_multi_alpha()
result = run_pipeline(config)

# Risk parity portfolio
config = PresetConfigs.risk_parity_portfolio()
result = run_pipeline(config)
```

### 2. Run from Configuration File

```python
from pipeline.config import PipelineConfig
from pipeline.orchestrator import run_pipeline

# Load from JSON
config = PipelineConfig.from_file("pipelines/simple_ema_cross.json")
result = run_pipeline(config)

# Load from YAML
config = PipelineConfig.from_yaml("pipelines/custom_strategy.yaml")
result = run_pipeline(config)
```

### 3. Create Custom Configuration

```python
from pipeline.config import (
    PipelineConfig, AlphaConfig, StrategyConfig,
    PortfolioConfig, BacktestConfig, AlphaType, 
    StrategyType, PortfolioAlgorithm
)

config = PipelineConfig(
    name="Custom Strategy",
    
    alpha=[
        AlphaConfig(type=AlphaType.EMA),
        AlphaConfig(type=AlphaType.RSI),
    ],
    
    strategies=[
        StrategyConfig(
            type=StrategyType.EMA_CROSS,
            alpha_factors=[AlphaConfig(type=AlphaType.EMA)]
        ),
    ],
    
    portfolio=PortfolioConfig(
        algorithm=PortfolioAlgorithm.ONS,
        strategies=[StrategyConfig(type=StrategyType.EMA_CROSS)],
        strategy_weights={"ema_cross": 1.0},
        params={"eta": 0.0, "beta": 1.0, "delta": 0.125}
    ),
    
    backtest=BacktestConfig(
        timerange="20250101-20250601",
        timeframe="4h",
        pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        initial_capital=50000.0
    ),
)

result = run_pipeline(config)
```

## Architecture

### Core Modules

#### `config.py` - Configuration Management
- **`PipelineConfig`** - Top-level configuration container
- **`AlphaConfig`** - Alpha factor configuration
- **`StrategyConfig`** - Trading strategy configuration
- **`PortfolioConfig`** - Portfolio optimization configuration
- **`BacktestConfig`** - Backtesting parameters
- **`DataConfig`** - Data source configuration
- **`PresetConfigs`** - Pre-built configurations for common scenarios

#### `orchestrator.py` - Pipeline Execution
- **`PortfolioPipeline`** - Main orchestrator class
- **`run_pipeline()`** - Convenience function

Pipeline stages:
1. Load OHLCV data
2. Generate alpha signals
3. Compute strategy signals
4. Construct portfolio weights
5. Run backtest
6. Compute metrics and validation

#### `verification.py` - Validation Framework
- **`PipelineVerification`** - Validation orchestrator
- **`ValidationResult`** - Individual validation result

Validation checks:
- **Data integrity:** pairs loaded, completeness, alignment, OHLC relationships, volume
- **Alpha signals:** required columns, no NaN, signal ranges
- **Strategy signals:** binary signals, signal density
- **Portfolio weights:** sum to 1.0, bounds [0,1], concentration
- **Backtest results:** required columns, portfolio value validity, return statistics

#### `results.py` - Result Aggregation
- **`PipelineResult`** - Complete execution result
- **`StageOutput`** - Individual stage output

Result features:
- Stage-by-stage tracking (duration, status, errors, warnings)
- Data preservation (prices, weights, backtest results)
- Metrics computation and aggregation
- Multiple export formats (JSON, CSV, HTML)
- HTML report generation

#### `integrations.py` - Freqtrade Integration
- **`FreqtradeIntegration`** - Convert to/from freqtrade format
- **`BatchPipelineRunner`** - Run multiple pipelines
- **`PresetPipelineRunner`** - Convenience runners
- **`FreqtradeStrategyExporter`** - Export as IStrategy
- **`PipelineComparator`** - Compare results

## Configuration Format

### JSON Configuration

```json
{
  "name": "My Strategy",
  "version": "1.0",
  "description": "Strategy description",
  
  "alpha": [
    {
      "type": "ema",
      "params": {"fast_period": 12, "slow_period": 26}
    }
  ],
  
  "strategies": [
    {
      "type": "ema_cross",
      "alpha_factors": [{"type": "ema"}]
    }
  ],
  
  "portfolio": {
    "algorithm": "ons",
    "strategies": [{"type": "ema_cross"}],
    "strategy_weights": {"ema_cross": 1.0},
    "params": {"eta": 0.0, "beta": 1.0}
  },
  
  "backtest": {
    "timerange": "20250101-20250601",
    "timeframe": "4h",
    "pairs": ["BTC/USDT", "ETH/USDT"],
    "initial_capital": 10000.0
  },
  
  "data": {
    "data_dir": "./user_data/data/usstock",
    "exchange": "portfoliobench"
  },
  
  "enable_validation": true,
  "output_dir": "./output"
}
```

### YAML Configuration

```yaml
name: "My Strategy"
version: "1.0"

alpha:
  - type: ema
    params:
      fast_period: 12
      slow_period: 26

strategies:
  - type: ema_cross
    alpha_factors:
      - type: ema

portfolio:
  algorithm: ons
  strategies:
    - type: ema_cross
  strategy_weights:
    ema_cross: 1.0
  params:
    eta: 0.0
    beta: 1.0

backtest:
  timerange: "20250101-20250601"
  timeframe: "4h"
  pairs:
    - BTC/USDT
    - ETH/USDT
  initial_capital: 10000.0

data:
  data_dir: "./user_data/data/usstock"
  exchange: "portfoliobench"

enable_validation: true
output_dir: "./output"
```

## Available Components

### Alpha Factors

| Type | Indicators | Configuration |
|------|-----------|---|
| **ema** | EMA fast/slow/exit, mean-volume | `fast_period`, `slow_period`, `exit_period` |
| **rsi** | RSI, signal line, overbought/oversold | `period`, `overbought`, `oversold` |
| **macd** | MACD, signal, histogram | `fast_period`, `slow_period`, `signal_period` |
| **bollinger** | Bands, bandwidth, %B | `period`, `std_dev` |
| **polymarket** | Probability, momentum, volume | Parameters vary |

### Strategy Types

| Type | Entry Signal | Exit Signal | Alpha Factors |
|------|------|------|------|
| **ema_cross** | EMA fast > EMA slow + volume filter | EMA exit < EMA fast | EMA |
| **rsi_bollinger** | RSI + Bollinger Bands | RSI + Bollinger Bands reversal | RSI |
| **macd_adx** | MACD > signal + ADX > 25 | MACD < signal + ADX > 25 | MACD |
| **ichimoku** | Ichimoku Cloud signals | Ichimoku Cloud reversal | - |
| **stochastic_cci** | Stochastic + CCI | Stochastic + CCI reversal | - |
| **mlp_speculative** | MLP model predictions | MLP model predictions | - |
| **polymarket_momentum** | Contract momentum | Momentum exit | Polymarket |
| **polymarket_mean_reversion** | Contract reversion | Reversion exit | Polymarket |

### Portfolio Algorithms

| Algorithm | Method | Rebalance | Parameters |
|------|--------|-----------|---|
| **ons** | Online Newton Step convex optimization | Per-candle | `eta`, `beta`, `delta` |
| **inverse_volatility** | Weight ∝ 1/volatility | Monthly | `lookback_window` |
| **min_variance** | Minimize portfolio variance | Monthly | `target_vol` |
| **best_single_asset** | Momentum rotation (winner-takes-all) | Monthly | `lookback_window` |
| **exponential_gradient** | Multiplicative weight update | Per-candle | `learning_rate` |
| **max_sharpe** | Maximize Sharpe ratio | Monthly | `target_vol` |
| **risk_parity** | Equal risk contribution | Monthly | `target_vol`, `min/max_weight` |
| **polymarket** | Prediction market weighted | Monthly | Parameters vary |
| **equal_weight** | 1/N allocation | Static | - |

## Usage Examples

### Example 1: Simple Pipeline from Configuration

```python
from pipeline.orchestrator import run_pipeline

result = run_pipeline("pipelines/simple_ema_cross.json")

# Access results
print(f"Total return: {result.metrics['total_return_pct']}%")
print(f"Sharpe ratio: {result.metrics['annualised_sharpe']:.4f}")

# Save outputs
result.save_all("./output/my_run")
```

### Example 2: Run and Compare Multiple Strategies

```python
from pipeline.integrations import (
    BatchPipelineRunner,
    PipelineComparator,
)

runner = BatchPipelineRunner()

configs = [
    "pipelines/simple_ema_cross.json",
    "pipelines/balanced_multi_alpha.json",
    "pipelines/risk_parity.json",
]

results = runner.run_multiple(configs)

# Compare results
PipelineComparator.print_comparison(results)
```

### Example 3: Export to Freqtrade

```python
from pipeline.integrations import FreqtradeIntegration

# Get result from pipeline
result = run_pipeline(config)

# Convert config to freqtrade format
ft_config = FreqtradeIntegration.config_from_pipeline(result.config)

# Export weights and backtest results
FreqtradeIntegration.export_backtest_results_for_freqtrade(
    result.backtest_result,
    result.portfolio_weights,
    "./output/freqtrade_export"
)
```

### Example 4: Programmatic Configuration

```python
from pipeline.config import (
    PipelineConfig, AlphaConfig, StrategyConfig,
    PortfolioConfig, BacktestConfig
)
from pipeline.orchestrator import PortfolioPipeline

config = PipelineConfig(
    name="Custom Strategy",
    alpha=[AlphaConfig(type="ema"), AlphaConfig(type="rsi")],
    strategies=[
        StrategyConfig(type="ema_cross", alpha_factors=[AlphaConfig(type="ema")])
    ],
    portfolio=PortfolioConfig(
        algorithm="ons",
        strategy_weights={"ema_cross": 1.0}
    ),
    backtest=BacktestConfig(
        timerange="20250101-20250601",
        pairs=["BTC/USDT", "ETH/USDT", "AAPL/USD"]
    )
)

pipeline = PortfolioPipeline(config, verbose=True)
result = pipeline.run()
result.print_summary()
```

## Output

### Pipeline Result Structure

```
result.
├── pipeline_name: str
├── start_time: str (ISO format)
├── end_time: str (ISO format)
├── duration_s: float
├── stages: Dict[str, StageOutput]
├── pair_data: pd.DataFrame (prices)
├── enriched_data: Dict[str, pd.DataFrame] (with alpha signals)
├── strategy_signals: Dict[str, pd.Series]
├── portfolio_weights: pd.DataFrame
├── backtest_result: pd.DataFrame (dates, values, returns)
├── metrics: Dict[str, float] (return, sharpe, drawdown, etc.)
├── validation: Dict[str, Any] (validation summary)
└── config: Dict[str, Any] (configuration used)
```

### Saved Outputs

**JSON Report:**
```json
{
  "pipeline_name": "Simple EMA Cross",
  "start_time": "2026-01-15T10:30:00",
  "end_time": "2026-01-15T10:35:45",
  "duration_s": 345.2,
  "stages": {...},
  "metrics": {
    "total_return_pct": 23.45,
    "annualised_return_pct": 18.92,
    "annualised_sharpe": 1.2345,
    "max_drawdown_pct": -8.73
  },
  "validation": {
    "total": 15,
    "passed": 14,
    "failed": 1,
    "pass_rate": 93.3
  }
}
```

**HTML Report:** Auto-generated with metrics dashboard

**CSV Files:**
- `portfolio_weights.csv` - Weight matrix
- `backtest_results.csv` - Daily portfolio values and returns

## Validation

The pipeline includes comprehensive validation at each stage:

```python
# Enable all validation checks
config.enable_validation = True
config.validate_data_integrity = True
config.validate_alpha_signals = True
config.validate_strategy_signals = True
config.validate_portfolio_weights = True

result = run_pipeline(config)

# View validation results
print(result.validation["pass_rate"])  # 0-100%
for check in result.validation["results"]:
    print(f"{check['name']}: {check['passed']}")
```

Available checks:
- Data completeness and alignment
- OHLC relationship validity
- Alpha signal ranges
- Strategy signal density
- Portfolio weight constraints
- Backtest result validity

## CLI Integration

Run pipelines from command line:

```bash
# Run from configuration
portbench pipeline pipelines/simple_ema_cross.json

# Run with output directory
portbench pipeline pipelines/balanced_multi_alpha.json --output output/results

# Run with verbose logging
portbench pipeline config.yaml --verbose

# Compare multiple pipelines
portbench pipeline-batch pipelines/ --compare
```

## Performance Considerations

- **Data loading:** Cached after first run
- **Alpha computation:** Uses TA-Lib (vendored)
- **Portfolio optimization:** Uses NumPy/SciPy
- **Backtesting:** Vectorized with Pandas

For large date ranges or many assets, consider:
1. Filtering to relevant timeframes
2. Reducing rebalance frequency
3. Using simpler alpha factors
4. Running in parallel with `BatchPipelineRunner`

## Troubleshooting

### Data Not Found

```python
# Check data directory
config.data.data_dir = "./user_data/data/usstock"
```

### Missing AlphaTaLib

```bash
pip install ta-lib
# or use vendored version in requirements-ml.txt
```

### Validation Failures

```python
# Enable verbose logging
config.verbose = True

# Check validation report
result = run_pipeline(config)
print(result.validation)
```

### Performance Issues

```python
# Reduce computation scope
config.backtest.timeframe = "1d"  # Fewer candles
config.backtest.timerange = "20250101-20250201"  # Shorter range
```

## Advanced Topics

### Custom Alpha Factors

See [alpha/interface.py](../alpha/interface.py) for `IAlpha` base class.

### Custom Strategies

See [strategy/](../strategy/) for strategy implementations.

### Custom Portfolio Algorithms

See [user_data/strategies/](../user_data/strategies/) for portfolio strategy templates.

### Freqtrade Dashboard Integration

Export results and visualize in freqtrade dashboard:

```python
from pipeline.integrations import FreqtradeStrategyExporter

FreqtradeStrategyExporter.export_to_strategy_file(
    result,
    "./strategy/PortfolioBench_Strategy.py"
)
```

## API Reference

See inline docstrings and examples in:
- `pipeline/config.py` - Configuration objects
- `pipeline/orchestrator.py` - Pipeline execution
- `pipeline/verification.py` - Validation framework
- `pipeline/results.py` - Result handling
- `pipeline/integrations.py` - Integrations

## License

Part of PortfolioBench. See LICENSE file for details.

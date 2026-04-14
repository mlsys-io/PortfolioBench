"""
examples/pipeline_examples.py
=============================
Example usage of the PortfolioBench pipeline system.

This file demonstrates:
  1. Running a pipeline from configuration
  2. Creating custom pipeline configurations
  3. Using preset configurations
  4. Comparing multiple pipelines
  5. Exporting results to freqtrade format
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import PipelineConfig, PresetConfigs
from pipeline.integrations import (
    BatchPipelineRunner,
    FreqtradeIntegration,
    PipelineComparator,
    PresetPipelineRunner,
)
from pipeline.orchestrator import run_pipeline

# ============================================================================
# HELPER: Generate synthetic test data
# ============================================================================

def generate_synthetic_data(pairs: list, timeframe: str = "4h", days: int = 100) -> dict:
    """Generate synthetic OHLCV data for testing."""
    data = {}
    
    for pair in pairs:
        # Generate random price series
        dates = pd.date_range(end="2025-06-01", periods=days, freq="D")
        
        # Simulate realistic OHLCV
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, days)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, days)))
        open_ = close * (1 + np.random.normal(0, 0.005, days))
        volume = np.random.uniform(1e6, 1e8, days)
        
        df = pd.DataFrame({
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
        
        data[pair] = df
    
    return data



# ============================================================================
# EXAMPLE 1: Run a pipeline from a configuration file
# ============================================================================

def example_1_run_from_config():
    """Run a pipeline from a JSON configuration file."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Run pipeline from configuration file")
    print("=" * 70)
    
    # Load configuration from file
    config_file = PROJECT_ROOT / "pipelines" / "simple_ema_cross.json"
    config = PipelineConfig.from_file(config_file)
    
    # Run the pipeline
    result = run_pipeline(config)
    
    # Print summary
    result.print_summary()
    
    return result


# ============================================================================
# EXAMPLE 2: Create a custom pipeline configuration
# ============================================================================

def example_2_custom_config():
    """Create and run a custom pipeline configuration."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Create and run custom configuration")
    print("=" * 70)
    
    from pipeline.config import (
        AlphaConfig,
        AlphaType,
        BacktestConfig,
        DataConfig,
        PortfolioAlgorithm,
        PortfolioConfig,
        StrategyConfig,
        StrategyType,
    )
    
    # Build configuration programmatically
    config = PipelineConfig(
        name="Custom Strategy",
        description="Custom pipeline built from scratch",
        
        alpha=[
            AlphaConfig(type=AlphaType.EMA, params={}),
            AlphaConfig(type=AlphaType.MACD, params={}),
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
            params={"eta": 0.1, "beta": 0.5, "delta": 0.25}
        ),
        
        backtest=BacktestConfig(
            timerange="20250101-20250301",
            timeframe="4h",
            pairs=["BTC/USDT", "ETH/USDT", "AAPL/USD"],
            initial_capital=25000.0
        ),
        
        data=DataConfig(exchange="portfoliobench"),
        
        output_dir="./output/custom_strategy"
    )
    
    # Run the pipeline
    result = run_pipeline(config)
    return result


# ============================================================================
# EXAMPLE 3: Use preset configurations
# ============================================================================

def example_3_preset_configs():
    """Run multiple preset configurations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Run preset configurations")
    print("=" * 70)
    
    # Simple EMA cross
    print("\nRunning Simple EMA Cross...")
    result1 = PresetPipelineRunner.run_simple_ema_cross(
        pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        output_dir="./output/preset_ema"
    )
    
    # Balanced multi-alpha
    print("\nRunning Balanced Multi-Alpha...")
    result2 = PresetPipelineRunner.run_balanced_multi_alpha(
        pairs=["BTC/USDT", "ETH/USDT", "AAPL/USD", "MSFT/USD"],
        output_dir="./output/preset_multi"
    )
    
    # Risk parity
    print("\nRunning Risk Parity...")
    result3 = PresetPipelineRunner.run_risk_parity(
        pairs=["BTC/USDT", "ETH/USDT", "SPY/USD", "TLT/USD"],
        output_dir="./output/preset_riskparity"
    )
    
    return [result1, result2, result3]


# ============================================================================
# EXAMPLE 4: Compare multiple pipelines
# ============================================================================

def example_4_compare_pipelines():
    """Run and compare multiple pipeline configurations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Compare multiple pipelines")
    print("=" * 70)
    
    runner = BatchPipelineRunner()
    
    # Run multiple configurations
    config_files = [
        PROJECT_ROOT / "pipelines" / "simple_ema_cross.json",
        PROJECT_ROOT / "pipelines" / "balanced_multi_alpha.json",
        PROJECT_ROOT / "pipelines" / "risk_parity.json",
    ]
    
    results = runner.run_multiple(config_files, output_dir="./output/batch_run")
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    PipelineComparator.print_comparison([r for r in results if r is not None])
    
    # Print summary
    summary = runner.get_summary()
    print("\nBatch Summary:")
    print(f"  Total runs: {summary['total_runs']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Failed: {summary['failed']}")
    
    return results


# ============================================================================
# EXAMPLE 5: Export results to freqtrade format
# ============================================================================

def example_5_export_freqtrade(pipeline_result):
    """Export pipeline results to freqtrade format."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Export to freqtrade format")
    print("=" * 70)
    
    output_dir = Path("./output/freqtrade_export")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export weights
    if pipeline_result.portfolio_weights is not None:
        weights_file = output_dir / "portfolio_weights.pkl"
        FreqtradeIntegration.export_weights_for_freqtrade(
            pipeline_result.portfolio_weights,
            weights_file
        )
    
    # Export backtest results
    if pipeline_result.backtest_result is not None:
        ft_files = FreqtradeIntegration.export_backtest_results_for_freqtrade(
            pipeline_result.backtest_result,
            pipeline_result.portfolio_weights,
            output_dir
        )
        print(f"Exported files: {ft_files}")
    
    # Generate freqtrade config
    ft_config = FreqtradeIntegration.config_from_pipeline(pipeline_result.config)
    print(f"\nFreqtrade config:\n{ft_config}")


# ============================================================================
# EXAMPLE 6: Pipeline with validation
# ============================================================================

def example_6_with_validation():
    """Run pipeline with comprehensive validation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Pipeline with validation")
    print("=" * 70)
    
    config = PresetConfigs.simple_ema_cross()
    config.enable_validation = True
    config.validate_data_integrity = True
    config.validate_alpha_signals = True
    config.validate_strategy_signals = True
    config.validate_portfolio_weights = True
    config.verbose = True
    
    # Generate synthetic test data since real data may not be available
    print("\nGenerating synthetic test data...")
    synthetic_data = generate_synthetic_data(config.backtest.pairs, days=100)
    
    # Save synthetic data to cache directory
    cache_dir = Path(config.data.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    for pair, df in synthetic_data.items():
        filename = pair.replace("/", "_") + f"-{config.backtest.timeframe}.feather"
        filepath = cache_dir / filename
        df.to_feather(filepath)
    
    # Update config to use cache directory
    config.data.data_dir = str(cache_dir)
    
    result = run_pipeline(config)
    
    # Print validation details
    if result.validation:
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total checks: {result.validation.get('total', 0)}")
        print(f"Passed: {result.validation.get('passed', 0)}")
        print(f"Failed: {result.validation.get('failed', 0)}")
        print(f"Pass rate: {result.validation.get('pass_rate', 0):.1f}%")
    
    return result


# ============================================================================
# MAIN: Run all examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PortfolioBench Pipeline Examples")
    print("=" * 70)
    
    try:
        # Uncomment to run individual examples:
        
        # Example 1: Run from config file
        # result1 = example_1_run_from_config()
        
        # Example 2: Create custom config
        # result2 = example_2_custom_config()
        
        # Example 3: Use preset configs
        # results3 = example_3_preset_configs()
        
        # Example 4: Compare pipelines
        # results4 = example_4_compare_pipelines()
        
        # Example 5: Export to freqtrade (requires result from another example)
        # example_5_export_freqtrade(result1)
        
        # Example 6: Run with validation
        result6 = example_6_with_validation()
        
        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

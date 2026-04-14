"""
pipeline/integrations.py
========================
Integration examples with freqtrade, backtesting engines, and external systems.

Demonstrates how to:
  - Run pipelines from freqtrade CLI commands
  - Export pipeline outputs to freqtrade format
  - Integrate with freqtrade dashboard/UI
  - Run multiple pipelines in parallel
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 1. FREQTRADE INTEGRATION
# ============================================================================

class FreqtradeIntegration:
    """Bridge between PortfolioBench pipelines and freqtrade backtesting."""
    
    @staticmethod
    def config_from_pipeline(pipeline_config) -> Dict[str, Any]:
        """
        Convert a PipelineConfig to a freqtrade config dict.
        
        Parameters
        ----------
        pipeline_config : PipelineConfig
            PortfolioBench pipeline configuration
        
        Returns
        -------
        dict
            Freqtrade configuration
        """
        from pipeline.config import PipelineConfig
        
        if isinstance(pipeline_config, PipelineConfig):
            config_dict = pipeline_config.to_dict()
        else:
            config_dict = pipeline_config
        
        # Extract backtest settings
        backtest_cfg = config_dict.get("backtest", {})
        timerange = backtest_cfg.get("timerange", "")
        
        # Parse timerange
        if timerange and "-" in timerange:
            start, end = timerange.split("-")
        else:
            start, end = "", ""
        
        freqtrade_config = {
            "exchange": {
                "name": config_dict.get("data", {}).get("exchange", "portfoliobench"),
                "pair_whitelist": backtest_cfg.get("pairs", []),
            },
            "stake_currency": "USDT",
            "dry_run_wallet": backtest_cfg.get("dry_run_wallet", backtest_cfg.get("initial_capital", 10000)),
            "timeframe": backtest_cfg.get("timeframe", "1d"),
            "max_open_trades": len(backtest_cfg.get("pairs", [])),
            "datadir": config_dict.get("data", {}).get("data_dir", "./user_data/data/usstock"),
            "user_data_dir": "./user_data",
            "tradingmode": "spot",
            "margin_mode": "isolated",
        }
        
        # Add date range if available
        if start and end:
            freqtrade_config["datetime"] = {
                "timerange": timerange,
            }
        
        return freqtrade_config
    
    @staticmethod
    def export_weights_for_freqtrade(
        portfolio_weights: pd.DataFrame,
        output_file: str | Path,
    ) -> str:
        """
        Export portfolio weights in freqtrade-compatible format.
        
        Parameters
        ----------
        portfolio_weights : pd.DataFrame
            Portfolio weights (columns=pairs, rows=dates)
        output_file : str or Path
            Output file path
        
        Returns
        -------
        str
            Path to saved file
        """
        output_file = Path(output_file)
        
        # Save as pickled DataFrame (freqtrade format)
        with open(output_file, "wb") as f:
            pickle.dump(portfolio_weights, f)
        
        logger.info(f"Exported portfolio weights to {output_file}")
        return str(output_file)
    
    @staticmethod
    def export_backtest_results_for_freqtrade(
        backtest_result: pd.DataFrame,
        portfolio_weights: pd.DataFrame,
        output_dir: str | Path,
    ) -> Dict[str, str]:
        """
        Export backtest results in freqtrade-compatible format.
        
        Parameters
        ----------
        backtest_result : pd.DataFrame
            Backtest results (date, portfolio_value, daily_return)
        portfolio_weights : pd.DataFrame
            Portfolio weights used
        output_dir : str or Path
            Output directory
        
        Returns
        -------
        dict
            Paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Save backtest results as JSON
        backtest_dict = {
            "results": backtest_result.to_dict(orient="records"),
            "summary": {
                "total_return": float(
                    (backtest_result["portfolio_value"].iloc[-1] / backtest_result["portfolio_value"].iloc[0]) - 1
                ),
                "sharpe_ratio": float(backtest_result["daily_return"].mean() / backtest_result["daily_return"].std() * (252 ** 0.5)),
                "max_drawdown": float(
                    ((backtest_result["portfolio_value"].cummax() - backtest_result["portfolio_value"]) / 
                     backtest_result["portfolio_value"].cummax()).min()
                ),
            }
        }
        
        results_file = output_dir / "backtest_results.json"
        with open(results_file, "w") as f:
            json.dump(backtest_dict, f, indent=2, default=str)
        files["results_json"] = str(results_file)
        
        # Save weights
        weights_file = output_dir / "portfolio_weights.json"
        with open(weights_file, "w") as f:
            json.dump(portfolio_weights.to_dict(orient="index"), f, indent=2, default=str)
        files["weights_json"] = str(weights_file)
        
        logger.info(f"Exported backtest results to {output_dir}")
        return files


# ============================================================================
# 2. BATCH PIPELINE EXECUTION
# ============================================================================

class BatchPipelineRunner:
    """Run multiple pipelines sequentially or in parallel."""
    
    def __init__(self, num_workers: int = 1):
        """
        Parameters
        ----------
        num_workers : int
            Number of parallel workers (1 = sequential)
        """
        self.num_workers = num_workers
        self.results = []
    
    def run_multiple(
        self,
        configs: List[str | Path],
        output_dir: str | Path = "./output",
    ) -> List[Any]:
        """
        Run multiple pipeline configurations.
        
        Parameters
        ----------
        configs : list of str or Path
            Configuration file paths
        output_dir : str or Path
            Base output directory
        
        Returns
        -------
        list
            Pipeline results
        """
        from pipeline.orchestrator import run_pipeline
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, config_path in enumerate(configs):
            config_path = Path(config_path)
            config_name = config_path.stem
            stage_dir = output_dir / config_name
            
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running pipeline {i+1}/{len(configs)}: {config_name}")
            logger.info(f"{'=' * 70}")
            
            try:
                from pipeline.config import PipelineConfig
                cfg = PipelineConfig.from_file(config_path) if config_path.suffix == ".json" else PipelineConfig.from_yaml(config_path)
                cfg.output_dir = str(stage_dir)
                
                result = run_pipeline(cfg)
                self.results.append(result)
                results.append(result)
            except Exception as e:
                logger.error(f"Pipeline failed: {e}", exc_info=True)
                results.append(None)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline runs."""
        successful = [r for r in self.results if r is not None]
        failed = len(self.results) - len(successful)
        
        metrics_summary = {}
        for result in successful:
            metrics_summary[result.pipeline_name] = result.metrics
        
        return {
            "total_runs": len(self.results),
            "successful": len(successful),
            "failed": failed,
            "metrics": metrics_summary,
        }


# ============================================================================
# 3. PRESET PIPELINE RUNNERS
# ============================================================================

class PresetPipelineRunner:
    """Convenience runners for common pipeline scenarios."""
    
    @staticmethod
    def run_simple_ema_cross(
        pairs: List[str] = None,
        output_dir: str | Path = "./output",
    ) -> Any:
        """Run simple EMA cross strategy."""
        from pipeline.config import PresetConfigs
        from pipeline.orchestrator import run_pipeline
        
        config = PresetConfigs.simple_ema_cross(pairs)
        config.output_dir = str(output_dir)
        return run_pipeline(config)
    
    @staticmethod
    def run_balanced_multi_alpha(
        pairs: List[str] = None,
        output_dir: str | Path = "./output",
    ) -> Any:
        """Run balanced multi-alpha strategy."""
        from pipeline.config import PresetConfigs
        from pipeline.orchestrator import run_pipeline
        
        config = PresetConfigs.balanced_multi_alpha(pairs)
        config.output_dir = str(output_dir)
        return run_pipeline(config)
    
    @staticmethod
    def run_risk_parity(
        pairs: List[str] = None,
        output_dir: str | Path = "./output",
    ) -> Any:
        """Run risk parity portfolio."""
        from pipeline.config import PresetConfigs
        from pipeline.orchestrator import run_pipeline
        
        config = PresetConfigs.risk_parity_portfolio(pairs)
        config.output_dir = str(output_dir)
        return run_pipeline(config)


# ============================================================================
# 4. FREQTRADE STRATEGY EXPORT
# ============================================================================

class FreqtradeStrategyExporter:
    """Export pipelines as freqtrade trading strategies."""
    
    @staticmethod
    def export_to_strategy_file(
        pipeline_result: Any,
        output_file: str | Path,
        strategy_class_name: str = "PortfolioBenchStrategy",
    ) -> str:
        """
        Export pipeline as a freqtrade IStrategy implementation.
        
        Parameters
        ----------
        pipeline_result : PipelineResult
            Pipeline execution result
        output_file : str or Path
            Output strategy file path
        strategy_class_name : str
            Strategy class name
        
        Returns
        -------
        str
            Path to generated strategy file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate strategy code
        weights_df = pipeline_result.portfolio_weights
        metrics = pipeline_result.metrics
        
        strategy_code = _generate_strategy_code(
            strategy_class_name=strategy_class_name,
            weights_df=weights_df,
            metrics=metrics,
            config=pipeline_result.config,
        )
        
        output_file.write_text(strategy_code)
        logger.info(f"Exported strategy to {output_file}")
        return str(output_file)


def _generate_strategy_code(
    strategy_class_name: str,
    weights_df: pd.DataFrame,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Generate freqtrade strategy code from pipeline results."""
    
    # Average weights across time
    avg_weights = weights_df.mean()
    weights_str = ",\n            ".join(
        f'"{pair}": {weight:.4f}' for pair, weight in avg_weights.items()
    )
    
    # Get metrics
    pipeline_name = config.get("name", "Unknown")
    total_return = metrics.get("total_return_pct", 0)
    sharpe = metrics.get("annualised_sharpe", 0)
    max_dd = metrics.get("max_drawdown_pct", 0)
    
    template = '''"""
Auto-generated strategy from PortfolioBench pipeline.

Pipeline: {pipeline_name}
Total return: {total_return:.2f}%
Sharpe ratio: {sharpe:.4f}
Max drawdown: {max_dd:.2f}%
"""

from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta


class {strategy_class_name}(IStrategy):
    """
    Auto-generated strategy from PortfolioBench.
    
    Average portfolio weights:
    {weights_str}
    """
    
    # Buy hyperspace parameters:
    buy_params = {{}}
    
    # Sell hyperspace parameters:
    sell_params = {{}}
    
    # ROI table:
    minimal_roi = {{"0": 0.1}}
    
    # Stoploss:
    stoploss = -0.10
    
    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Candle timeframe
    timeframe = '1d'
    
    # Run the strategy analysis every N candles
    can_short = False
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Add technical indicators."""
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions."""
        dataframe.loc[:, 'enter_long'] = 0
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define exit conditions."""
        dataframe.loc[:, 'exit_long'] = 0
        return dataframe
'''
    
    return template.format(
        pipeline_name=pipeline_name,
        total_return=total_return,
        sharpe=sharpe,
        max_dd=max_dd,
        strategy_class_name=strategy_class_name,
        weights_str=weights_str
    )


# ============================================================================
# 5. COMPARATOR
# ============================================================================

class PipelineComparator:
    """Compare results from multiple pipelines."""
    
    @staticmethod
    def compare_results(results: List[Any]) -> pd.DataFrame:
        """
        Compare metrics across multiple pipeline results.
        
        Parameters
        ----------
        results : list of PipelineResult
            Pipeline results to compare
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        comparison_data = []
        
        for result in results:
            if result is None:
                continue
            
            comparison_data.append({
                "Pipeline": result.pipeline_name,
                "Duration (s)": round(result.duration_s, 2),
                "Total Return (%)": result.metrics.get("total_return_pct", 0),
                "Ann. Return (%)": result.metrics.get("annualised_return_pct", 0),
                "Sharpe Ratio": result.metrics.get("annualised_sharpe", 0),
                "Max Drawdown (%)": result.metrics.get("max_drawdown_pct", 0),
                "Validation Pass Rate (%)": result.validation.get("pass_rate", 0),
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def print_comparison(results: List[Any]) -> None:
        """Print formatted comparison."""
        df = PipelineComparator.compare_results(results)
        print("\n" + "=" * 100)
        print("PIPELINE COMPARISON")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

"""
pipeline/orchestrator.py
========================
Main pipeline orchestrator and execution engine.

Coordinates the complete portfolio construction workflow:
  1. Load OHLCV data
  2. Generate alpha factor signals
  3. Compute strategy signals
  4. Construct portfolio weights
  5. Run backtest
  6. Compute metrics and validation
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.config import PipelineConfig, AlphaType, StrategyType
from pipeline.verification import PipelineVerification, ValidationResult
from pipeline.results import PipelineResult, StageOutput

from portfolio.PortfolioManagement import (
    load_pair_data,
    align_close_prices,
    ema_cross_signals,
    build_ema_position_series,
    calculate_ons_weights,
    equal_weight_allocation,
    blend_strategy_weights,
    backtest_portfolio,
    compute_metrics,
)

logger = logging.getLogger(__name__)


class PortfolioPipeline:
    """End-to-end portfolio construction and backtesting pipeline."""
    
    def __init__(self, config: PipelineConfig, verbose: bool = False):
        """
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        verbose : bool
            Enable verbose logging
        """
        self.config = config
        self.verbose = verbose or config.verbose
        self.result = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
    
    def run(self) -> PipelineResult:
        """Execute the complete pipeline."""
        start_time = time.time()
        start_dt = pd.Timestamp.now()
        
        # Initialize result object
        self.result = PipelineResult(
            pipeline_name=self.config.name,
            start_time=start_dt.isoformat(),
            end_time="",
            duration_s=0.0,
            config=self.config.to_dict()
        )
        
        verification = PipelineVerification(
            strict=False,
            verbose=self.verbose
        )
        
        try:
            # Stage 1: Load data
            logger.info("=" * 70)
            logger.info("STAGE 1: Loading OHLCV Data")
            logger.info("=" * 70)
            pair_data = self._stage_load_data()
            self.result.add_stage_output("load_data", StageOutput(
                name="load_data",
                status="success",
                duration_s=0,
                data_summary={"pairs_loaded": len(pair_data)}
            ))
            
            # Stage 2: Generate alpha signals
            if self.config.alpha:
                logger.info("\n" + "=" * 70)
                logger.info("STAGE 2: Generating Alpha Signals")
                logger.info("=" * 70)
                enriched_data = self._stage_generate_alpha(pair_data)
                self.result.enriched_data = enriched_data
                self.result.add_stage_output("generate_alpha", StageOutput(
                    name="generate_alpha",
                    status="success",
                    duration_s=0,
                    data_summary={"alphas_applied": len(self.config.alpha)}
                ))
                
                # Validate alpha signals
                if self.config.validate_alpha_signals:
                    expected_cols = self._get_expected_alpha_columns()
                    verification.validate_alpha_signals(enriched_data, expected_cols)
            else:
                enriched_data = pair_data
                logger.info("No alpha factors configured, skipping alpha generation")
            
            # Stage 3: Compute strategy signals
            if self.config.strategies:
                logger.info("\n" + "=" * 70)
                logger.info("STAGE 3: Computing Strategy Signals")
                logger.info("=" * 70)
                strategy_signals = self._stage_compute_strategy_signals(enriched_data)
                self.result.strategy_signals = strategy_signals
                self.result.add_stage_output("compute_strategy_signals", StageOutput(
                    name="compute_strategy_signals",
                    status="success",
                    duration_s=0,
                    data_summary={"strategies": len(self.config.strategies)}
                ))
                
                # Validate strategy signals
                if self.config.validate_strategy_signals:
                    verification.validate_strategy_signals(strategy_signals)
            else:
                logger.warning("No strategies configured, skipping strategy signal computation")
            
            # Stage 4: Construct portfolio weights
            if self.config.portfolio:
                logger.info("\n" + "=" * 70)
                logger.info("STAGE 4: Constructing Portfolio Weights")
                logger.info("=" * 70)
                prices = align_close_prices(pair_data)
                weights = self._stage_construct_portfolio(prices, enriched_data)
                self.result.portfolio_weights = weights
                self.result.add_stage_output("construct_portfolio", StageOutput(
                    name="construct_portfolio",
                    status="success",
                    duration_s=0,
                    data_summary={"assets": weights.shape[1], "rows": weights.shape[0]}
                ))
                
                # Validate portfolio weights
                if self.config.validate_portfolio_weights:
                    verification.validate_portfolio_weights(weights)
            else:
                logger.warning("No portfolio configuration, skipping portfolio construction")
            
            # Stage 5: Run backtest
            if self.config.portfolio and self.config.backtest:
                logger.info("\n" + "=" * 70)
                logger.info("STAGE 5: Running Backtest")
                logger.info("=" * 70)
                backtest_result = self._stage_backtest(prices, weights)
                self.result.backtest_result = backtest_result
                
                metrics = compute_metrics(backtest_result)
                self.result.metrics = metrics
                self.result.add_stage_output("backtest", StageOutput(
                    name="backtest",
                    status="success",
                    duration_s=0,
                    data_summary=metrics
                ))
                
                logger.info(f"\nBacktest Results:")
                logger.info(f"  Total return:        {metrics['total_return_pct']:.2f}%")
                logger.info(f"  Annualized return:   {metrics['annualised_return_pct']:.2f}%")
                logger.info(f"  Annualized Sharpe:   {metrics['annualised_sharpe']:.4f}")
                logger.info(f"  Max drawdown:        {metrics['max_drawdown_pct']:.2f}%")
            
            # Store validation results
            if self.config.validate_data_integrity:
                verification.validate_data_integrity(pair_data, self.config.backtest.pairs if self.config.backtest else [])
            
            self.result.validation = verification.get_summary()
            
            # Print validation report
            if self.config.enable_validation:
                logger.info("\n" + verification.report())
            
            # Save outputs
            self._save_outputs()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self.result.add_stage_output("error", StageOutput(
                name="error",
                status="failure",
                duration_s=0,
                errors=[str(e)]
            ))
            raise
        
        finally:
            # Finalize result
            end_time = time.time()
            end_dt = pd.Timestamp.now()
            self.result.end_time = end_dt.isoformat()
            self.result.duration_s = end_time - start_time
            
            # Print summary
            self.result.print_summary()
        
        return self.result
    
    def _stage_load_data(self) -> Dict[str, pd.DataFrame]:
        """Load OHLCV data."""
        if not self.config.backtest:
            raise ValueError("No backtest configuration provided")
        
        pairs = self.config.backtest.pairs
        timeframe = self.config.backtest.timeframe
        data_dir = self.config.data.data_dir
        
        logger.info(f"Loading data for {len(pairs)} pairs...")
        pair_data = load_pair_data(data_dir, pairs, timeframe)
        
        if len(pair_data) == 0:
            raise RuntimeError("No data loaded - check data directory and pair names")
        
        return pair_data
    
    def _stage_generate_alpha(self, pair_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate alpha factor signals."""
        enriched = {}
        
        for pair, df in pair_data.items():
            df = df.copy()
            
            for alpha_cfg in self.config.alpha:
                alpha_type = alpha_cfg.type
                params = alpha_cfg.params
                
                if alpha_type == AlphaType.EMA:
                    from alpha.SimpleEmaFactors import EmaAlpha
                    alpha = EmaAlpha(df, metadata={"pair": pair})
                    df = alpha.process()
                    logger.debug(f"Applied EMA alpha to {pair}")
                
                elif alpha_type == AlphaType.RSI:
                    from alpha.RsiAlpha import RsiAlpha
                    alpha = RsiAlpha(df, metadata={"pair": pair})
                    df = alpha.process()
                    logger.debug(f"Applied RSI alpha to {pair}")
                
                elif alpha_type == AlphaType.MACD:
                    from alpha.MacdAlpha import MacdAlpha
                    alpha = MacdAlpha(df, metadata={"pair": pair})
                    df = alpha.process()
                    logger.debug(f"Applied MACD alpha to {pair}")
                
                elif alpha_type == AlphaType.BOLLINGER:
                    from alpha.BollingerAlpha import BollingerAlpha
                    alpha = BollingerAlpha(df, metadata={"pair": pair})
                    df = alpha.process()
                    logger.debug(f"Applied Bollinger alpha to {pair}")
                
                else:
                    logger.warning(f"Unknown alpha type: {alpha_type}")
            
            enriched[pair] = df
        
        logger.info(f"Generated alpha signals for {len(enriched)} pairs")
        return enriched
    
    def _stage_compute_strategy_signals(self, enriched_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Compute trading strategy signals."""
        strategy_signals = {}
        
        for strategy_cfg in self.config.strategies:
            strategy_type = strategy_cfg.type
            
            for pair, df in enriched_data.items():
                if strategy_type == StrategyType.EMA_CROSS:
                    df_signals = ema_cross_signals(df)
                    pos = build_ema_position_series(df_signals)
                    pos.index = df_signals["date"]
                    key = f"{pair}_{strategy_type.value}"
                    strategy_signals[key] = pos
                    logger.debug(f"Applied EMA Cross to {pair}: {pos.sum()} signals")
                
                elif strategy_type == StrategyType.RSI_BOLLINGER:
                    # RSI Bollinger strategy signals
                    if "rsi_oversold" in df.columns and "rsi_overbought" in df.columns:
                        df = df.copy()
                        df["enter_long"] = ((df["rsi_oversold"] == 1) & (df.get("mean-volume", 1) > 0.75)).astype(int)
                        df["exit_long"] = (df["rsi_overbought"] == 1).astype(int)
                        pos = build_ema_position_series(df)
                        pos.index = df["date"]
                        key = f"{pair}_{strategy_type.value}"
                        strategy_signals[key] = pos
                        logger.debug(f"Applied RSI Bollinger to {pair}")
                
                # Add more strategy types as needed
        
        logger.info(f"Computed signals for {len(strategy_signals)} strategy-pair combinations")
        return strategy_signals
    
    def _stage_construct_portfolio(
        self,
        prices: pd.DataFrame,
        enriched_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Construct portfolio weights."""
        portfolio_cfg = self.config.portfolio
        pairs = list(prices.columns)
        
        # Compute ONS weights if using ONS algorithm
        if portfolio_cfg.algorithm.value == "ons":
            params = portfolio_cfg.params
            ons_weights = calculate_ons_weights(
                prices,
                eta=params.get("eta", 0.0),
                beta=params.get("beta", 1.0),
                delta=params.get("delta", 0.125)
            )
        else:
            # Default to equal weights for other algorithms
            ons_weights = pd.DataFrame(
                np.ones((len(prices), len(pairs))) / len(pairs),
                index=prices.index,
                columns=pairs
            )
        
        # Compute strategy positions
        ema_positions = {}
        for pair, df in enriched_data.items():
            if "enter_long" in df.columns and "exit_long" in df.columns:
                df_signals = ema_cross_signals(df)
                pos = build_ema_position_series(df_signals)
                pos.index = df_signals["date"]
                ema_positions[pair] = pos
        
        if not ema_positions:
            # Use equal weight if no positions computed
            for pair in pairs:
                ema_positions[pair] = pd.Series(1, index=prices.index, name=pair)
        
        # Compute equal weights
        equal_wt = equal_weight_allocation(pairs)
        
        # Blend strategies
        final_weights = blend_strategy_weights(
            ons_weights=ons_weights,
            ema_positions=ema_positions,
            equal_wt=equal_wt,
            w_equal=portfolio_cfg.strategy_weights.get("equal", 0.34),
            w_ons=portfolio_cfg.strategy_weights.get("ons", 0.33),
            w_ema=portfolio_cfg.strategy_weights.get("ema", 0.33),
        )
        
        logger.info(f"Constructed portfolio weights: {final_weights.shape[0]} rows × {final_weights.shape[1]} assets")
        return final_weights
    
    def _stage_backtest(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run portfolio backtest."""
        initial_capital = self.config.backtest.initial_capital
        
        logger.info(f"Running backtest with initial capital: ${initial_capital:,.0f}")
        result = backtest_portfolio(prices, weights, initial_capital)
        
        logger.info(f"Backtest complete: {len(result)} bars")
        return result
    
    def _get_expected_alpha_columns(self) -> List[str]:
        """Get list of columns expected from alpha factors."""
        columns = []
        
        for alpha in self.config.alpha:
            if alpha.type == AlphaType.EMA:
                columns.extend(["ema_fast", "ema_slow", "ema_exit", "mean-volume"])
            elif alpha.type == AlphaType.RSI:
                columns.extend(["rsi", "rsi_signal", "rsi_oversold", "rsi_overbought"])
            elif alpha.type == AlphaType.MACD:
                columns.extend(["macd", "macd_signal", "macd_hist", "macd_hist_rising"])
            elif alpha.type == AlphaType.BOLLINGER:
                columns.extend(["bb_upper", "bb_middle", "bb_lower", "bb_bandwidth", "bb_pct"])
        
        return list(set(columns))
    
    def _save_outputs(self) -> None:
        """Save pipeline outputs."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving outputs to {output_dir}")
        
        saved_files = self.result.save_all(output_dir)
        for key, filepath in saved_files.items():
            logger.info(f"  ✓ {key}: {filepath}")


# Convenience function
def run_pipeline(config: PipelineConfig | str | Path) -> PipelineResult:
    """
    Run a portfolio pipeline with the given configuration.
    
    Parameters
    ----------
    config : PipelineConfig, str, or Path
        Pipeline configuration (object, JSON file, or YAML file)
    
    Returns
    -------
    PipelineResult
        Complete pipeline execution result
    """
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        if config_path.suffix == ".yaml" or config_path.suffix == ".yml":
            config = PipelineConfig.from_yaml(config_path)
        else:
            config = PipelineConfig.from_file(config_path)
    
    pipeline = PortfolioPipeline(config, verbose=config.verbose if hasattr(config, 'verbose') else False)
    return pipeline.run()

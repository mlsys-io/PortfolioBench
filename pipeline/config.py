"""
pipeline/config.py
==================
Pipeline configuration management and validation.

Provides ConfigurationManagement, PipelineConfig, and ConfigLoader for
loading pipeline specifications from YAML, JSON, or Python dicts.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AlphaType(str, Enum):
    """Available alpha factor types."""
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER = "bollinger"
    POLYMARKET = "polymarket"


class StrategyType(str, Enum):
    """Available strategy types."""
    EMA_CROSS = "ema_cross"
    RSI_BOLLINGER = "rsi_bollinger"
    MACD_ADX = "macd_adx"
    ICHIMOKU = "ichimoku"
    STOCHASTIC_CCI = "stochastic_cci"
    MLP_SPECULATIVE = "mlp_speculative"
    POLYMARKET_MOMENTUM = "polymarket_momentum"
    POLYMARKET_MEAN_REVERSION = "polymarket_mean_reversion"


class PortfolioAlgorithm(str, Enum):
    """Available portfolio optimization algorithms."""
    ONS = "ons"
    INVERSE_VOLATILITY = "inverse_volatility"
    MIN_VARIANCE = "min_variance"
    BEST_SINGLE_ASSET = "best_single_asset"
    EXPONENTIAL_GRADIENT = "exponential_gradient"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    POLYMARKET = "polymarket"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class AlphaConfig:
    """Configuration for an alpha factor."""
    type: AlphaType
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = AlphaType(self.type)


@dataclass
class StrategyConfig:
    """Configuration for a trading strategy."""
    type: StrategyType
    alpha_factors: List[AlphaConfig] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = StrategyType(self.type)
        self.alpha_factors = [
            AlphaConfig(**a) if isinstance(a, dict) else a 
            for a in self.alpha_factors
        ]


@dataclass
class PortfolioConfig:
    """Configuration for portfolio construction."""
    algorithm: PortfolioAlgorithm
    strategies: List[StrategyConfig] = field(default_factory=list)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.algorithm, str):
            self.algorithm = PortfolioAlgorithm(self.algorithm)
        self.strategies = [
            StrategyConfig(**s) if isinstance(s, dict) else s 
            for s in self.strategies
        ]
        
        # Normalize strategy weights to sum to 1.0
        if self.strategy_weights:
            total = sum(self.strategy_weights.values())
            if total > 0:
                self.strategy_weights = {
                    k: v / total for k, v in self.strategy_weights.items()
                }


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    timerange: str  # "YYYYMMDD-YYYYMMDD"
    timeframe: str = "1d"  # 5m, 4h, 1d
    pairs: List[str] = field(default_factory=list)
    initial_capital: float = 10_000.0
    dry_run_wallet: Optional[float] = None
    enable_trading_mode: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data sources."""
    data_dir: str = "./user_data/data/usstock"
    exchange: str = "portfoliobench"  # portfoliobench, polymarket, binance
    cache_dir: str = "./user_data/cache"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    name: str
    version: str = "1.0"
    description: str = ""
    
    # Core components
    alpha: List[AlphaConfig] = field(default_factory=list)
    strategies: List[StrategyConfig] = field(default_factory=list)
    portfolio: Optional[PortfolioConfig] = None
    backtest: Optional[BacktestConfig] = None
    data: DataConfig = field(default_factory=DataConfig)
    
    # Verification and validation
    enable_validation: bool = True
    validate_data_integrity: bool = True
    validate_alpha_signals: bool = True
    validate_strategy_signals: bool = True
    validate_portfolio_weights: bool = True
    
    # Output and reporting
    output_dir: str = "./output"
    export_weights: bool = True
    export_backtest_results: bool = True
    export_metrics: bool = True
    verbose: bool = False
    
    def __post_init__(self):
        # Normalize nested configs
        self.alpha = [
            AlphaConfig(**a) if isinstance(a, dict) else a 
            for a in self.alpha
        ]
        self.strategies = [
            StrategyConfig(**s) if isinstance(s, dict) else s 
            for s in self.strategies
        ]
        if isinstance(self.portfolio, dict):
            self.portfolio = PortfolioConfig(**self.portfolio)
        if isinstance(self.backtest, dict):
            self.backtest = BacktestConfig(**self.backtest)
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save(self, filepath: str | Path) -> None:
        """Save config to JSON file."""
        Path(filepath).write_text(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """Create config from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> PipelineConfig:
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def from_file(cls, filepath: str | Path) -> PipelineConfig:
        """Load config from JSON file."""
        return cls.from_json(Path(filepath).read_text())
    
    @classmethod
    def from_yaml(cls, filepath: str | Path) -> PipelineConfig:
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML config support. "
                            "Install with: pip install PyYAML")
        
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Preset configurations for common use cases
class PresetConfigs:
    """Pre-built pipeline configurations."""
    
    @staticmethod
    def simple_ema_cross(pairs: List[str] = None) -> PipelineConfig:
        """Simple EMA cross strategy on crypto."""
        if pairs is None:
            pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        
        return PipelineConfig(
            name="Simple EMA Cross",
            description="Basic EMA crossover strategy on 4h timeframe",
            alpha=[
                AlphaConfig(type=AlphaType.EMA, params={})
            ],
            strategies=[
                StrategyConfig(
                    type=StrategyType.EMA_CROSS,
                    alpha_factors=[AlphaConfig(type=AlphaType.EMA)]
                )
            ],
            portfolio=PortfolioConfig(
                algorithm=PortfolioAlgorithm.EQUAL_WEIGHT,
                strategies=[StrategyConfig(type=StrategyType.EMA_CROSS)],
                strategy_weights={"EMA_CROSS": 1.0}
            ),
            backtest=BacktestConfig(
                timerange="20250101-20250601",
                timeframe="4h",
                pairs=pairs,
                initial_capital=10_000.0
            ),
            data=DataConfig(exchange="portfoliobench")
        )
    
    @staticmethod
    def balanced_multi_alpha(pairs: List[str] = None) -> PipelineConfig:
        """Balanced strategy using multiple alpha factors."""
        if pairs is None:
            pairs = ["BTC/USDT", "ETH/USDT", "SOL/USDT", 
                    "AAPL/USD", "MSFT/USD"]
        
        return PipelineConfig(
            name="Balanced Multi-Alpha",
            description="Balanced portfolio with EMA, RSI, and MACD alphas",
            alpha=[
                AlphaConfig(type=AlphaType.EMA),
                AlphaConfig(type=AlphaType.RSI),
                AlphaConfig(type=AlphaType.MACD),
            ],
            strategies=[
                StrategyConfig(
                    type=StrategyType.EMA_CROSS,
                    alpha_factors=[AlphaConfig(type=AlphaType.EMA)]
                ),
                StrategyConfig(
                    type=StrategyType.RSI_BOLLINGER,
                    alpha_factors=[AlphaConfig(type=AlphaType.RSI)]
                ),
                StrategyConfig(
                    type=StrategyType.MACD_ADX,
                    alpha_factors=[AlphaConfig(type=AlphaType.MACD)]
                ),
            ],
            portfolio=PortfolioConfig(
                algorithm=PortfolioAlgorithm.ONS,
                strategies=[
                    StrategyConfig(type=StrategyType.EMA_CROSS),
                    StrategyConfig(type=StrategyType.RSI_BOLLINGER),
                    StrategyConfig(type=StrategyType.MACD_ADX),
                ],
                strategy_weights={
                    "EMA_CROSS": 0.4,
                    "RSI_BOLLINGER": 0.3,
                    "MACD_ADX": 0.3,
                },
                params={"eta": 0.0, "beta": 1.0, "delta": 0.125}
            ),
            backtest=BacktestConfig(
                timerange="20250101-20250601",
                timeframe="1d",
                pairs=pairs,
                initial_capital=50_000.0
            ),
            data=DataConfig(exchange="portfoliobench")
        )
    
    @staticmethod
    def risk_parity_portfolio(pairs: List[str] = None) -> PipelineConfig:
        """Risk parity portfolio across multiple asset classes."""
        if pairs is None:
            pairs = ["BTC/USDT", "ETH/USDT", "AAPL/USD", 
                    "SPY/USD", "TLT/USD", "GLD/USD"]
        
        return PipelineConfig(
            name="Risk Parity Portfolio",
            description="Risk parity allocation across crypto, equities, bonds",
            alpha=[
                AlphaConfig(type=AlphaType.EMA),
            ],
            strategies=[
                StrategyConfig(
                    type=StrategyType.EMA_CROSS,
                    alpha_factors=[AlphaConfig(type=AlphaType.EMA)]
                ),
            ],
            portfolio=PortfolioConfig(
                algorithm=PortfolioAlgorithm.RISK_PARITY,
                strategies=[StrategyConfig(type=StrategyType.EMA_CROSS)],
                strategy_weights={"EMA_CROSS": 1.0},
                rebalance_frequency="monthly",
                params={"target_vol": 0.12}
            ),
            backtest=BacktestConfig(
                timerange="20240101-20260101",
                timeframe="1d",
                pairs=pairs,
                initial_capital=100_000.0
            ),
            data=DataConfig(exchange="portfoliobench")
        )

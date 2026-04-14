"""
tests/test_pipeline.py
======================
Unit and integration tests for the pipeline system.

Tests cover:
  - Configuration management
  - Pipeline execution
  - Validation framework
  - Result aggregation
  - Integrations
"""

import json

# Configure path
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.config import (
    AlphaConfig,
    AlphaType,
    BacktestConfig,
    PipelineConfig,
    PortfolioAlgorithm,
    PortfolioConfig,
    PresetConfigs,
    StrategyConfig,
    StrategyType,
)
from pipeline.integrations import FreqtradeIntegration, PipelineComparator
from pipeline.orchestrator import PortfolioPipeline
from pipeline.results import PipelineResult, StageOutput
from pipeline.verification import PipelineVerification, ValidationResult

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_prices_df():
    """Create sample price data."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1D")
    data = {
        "BTC/USDT": np.random.lognormal(10, 0.02, 100),
        "ETH/USDT": np.random.lognormal(8, 0.03, 100),
        "AAPL/USD": np.random.lognormal(6, 0.015, 100),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_portfolio_weights():
    """Create sample portfolio weights."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1D")
    n_assets = 3
    weights = np.random.dirichlet(np.ones(n_assets), 100)
    return pd.DataFrame(
        weights,
        index=dates,
        columns=["BTC/USDT", "ETH/USDT", "AAPL/USD"]
    )


@pytest.fixture
def sample_backtest_result():
    """Create sample backtest results."""
    dates = pd.date_range("2025-01-01", periods=100, freq="1D")
    portfolio_values = 10000 * (1 + np.random.normal(0.001, 0.02, 100)).cumprod()
    daily_returns = np.diff(portfolio_values, prepend=10000) / np.append(10000, portfolio_values[:-1]) - 1
    
    return pd.DataFrame({
        "date": dates,
        "portfolio_value": portfolio_values,
        "daily_return": daily_returns,
    })


@pytest.fixture
def simple_config():
    """Create a simple pipeline configuration."""
    return PipelineConfig(
        name="Test Pipeline",
        alpha=[AlphaConfig(type=AlphaType.EMA)],
        strategies=[StrategyConfig(type=StrategyType.EMA_CROSS)],
        portfolio=PortfolioConfig(algorithm=PortfolioAlgorithm.EQUAL_WEIGHT),
        backtest=BacktestConfig(
            timerange="20250101-20250331",
            pairs=["BTC/USDT", "ETH/USDT"],
            initial_capital=10000.0
        )
    )


# ============================================================================
# CONFIG TESTS
# ============================================================================

class TestPipelineConfig:
    """Test configuration management."""
    
    def test_config_creation(self):
        """Test creating configuration."""
        config = PipelineConfig(
            name="Test",
            alpha=[AlphaConfig(type=AlphaType.EMA)],
            backtest=BacktestConfig(
                timerange="20250101-20250201",
                pairs=["BTC/USDT"]
            )
        )
        
        assert config.name == "Test"
        assert len(config.alpha) == 1
        assert config.alpha[0].type == AlphaType.EMA
    
    def test_config_to_dict(self, simple_config):
        """Test config serialization to dict."""
        config_dict = simple_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "Test Pipeline"
        assert "alpha" in config_dict
        assert "backtest" in config_dict
    
    def test_config_to_json(self, simple_config):
        """Test config serialization to JSON."""
        json_str = simple_config.to_json()
        
        assert isinstance(json_str, str)
        assert "Test Pipeline" in json_str
        data = json.loads(json_str)
        assert data["name"] == "Test Pipeline"
    
    def test_config_save_and_load(self, simple_config):
        """Test saving and loading config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name
        
        try:
            # Save
            simple_config.save(temp_file)
            
            # Load
            loaded = PipelineConfig.from_file(temp_file)
            
            assert loaded.name == simple_config.name
            assert len(loaded.alpha) == len(simple_config.alpha)
        finally:
            Path(temp_file).unlink()
    
    def test_preset_configs(self):
        """Test preset configurations."""
        # Simple EMA
        config1 = PresetConfigs.simple_ema_cross()
        assert config1.name == "Simple EMA Cross"
        assert len(config1.alpha) == 1
        
        # Multi-alpha
        config2 = PresetConfigs.balanced_multi_alpha()
        assert config2.name == "Balanced Multi-Alpha"
        assert len(config2.alpha) == 3
        
        # Risk parity
        config3 = PresetConfigs.risk_parity_portfolio()
        assert config3.name == "Risk Parity Portfolio"
        assert config3.portfolio.algorithm == PortfolioAlgorithm.RISK_PARITY


# ============================================================================
# VERIFICATION TESTS
# ============================================================================

class TestPipelineVerification:
    """Test validation framework."""
    
    def test_validation_result(self):
        """Test validation result."""
        result = ValidationResult(
            name="test_check",
            passed=True,
            message="Check passed",
            details={"count": 5}
        )
        
        assert result.passed is True
        assert result.message == "Check passed"
        assert result.details["count"] == 5
    
    def test_validate_portfolio_weights(self, sample_portfolio_weights):
        """Test portfolio weight validation."""
        verifier = PipelineVerification()
        
        # Should pass for valid weights
        result = verifier.validate_portfolio_weights(sample_portfolio_weights)
        assert result is True
        
        # Check specific validation
        checks = [r for r in verifier.results if "weights_sum_to_one" in r.name]
        assert len(checks) > 0
        assert checks[0].passed is True
    
    def test_validate_invalid_weights(self):
        """Test validation with invalid weights."""
        # Create weights that don't sum to 1
        invalid_weights = pd.DataFrame(
            [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]],  # Don't sum to 1
            columns=["A", "B", "C"]
        )
        
        verifier = PipelineVerification()
        result = verifier.validate_portfolio_weights(invalid_weights)
        
        # Should fail
        assert result is False
    
    def test_get_validation_summary(self, sample_portfolio_weights):
        """Test validation summary."""
        verifier = PipelineVerification()
        verifier.validate_portfolio_weights(sample_portfolio_weights)
        
        summary = verifier.get_summary()
        
        assert "total" in summary
        assert "passed" in summary
        assert "failed" in summary
        assert "pass_rate" in summary
        assert "results" in summary


# ============================================================================
# RESULTS TESTS
# ============================================================================

class TestPipelineResults:
    """Test result aggregation."""
    
    def test_pipeline_result_creation(self):
        """Test creating pipeline result."""
        result = PipelineResult(
            pipeline_name="Test",
            start_time="2025-01-01T10:00:00",
            end_time="2025-01-01T10:05:00",
            duration_s=300.0
        )
        
        assert result.pipeline_name == "Test"
        assert result.duration_s == 300.0
    
    def test_add_stage_output(self):
        """Test adding stage output."""
        result = PipelineResult(
            pipeline_name="Test",
            start_time="2025-01-01T10:00:00",
            end_time="2025-01-01T10:05:00",
            duration_s=300.0
        )
        
        stage = StageOutput(
            name="load_data",
            status="success",
            duration_s=10.0,
            data_summary={"pairs": 3}
        )
        
        result.add_stage_output("load_data", stage)
        
        assert "load_data" in result.stages
        assert result.stages["load_data"].status == "success"
    
    def test_result_to_dict(self, sample_backtest_result):
        """Test result serialization."""
        result = PipelineResult(
            pipeline_name="Test",
            start_time="2025-01-01T10:00:00",
            end_time="2025-01-01T10:05:00",
            duration_s=300.0,
            metrics={
                "total_return_pct": 5.0,
                "annualised_sharpe": 1.5
            }
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["pipeline_name"] == "Test"
        assert result_dict["metrics"]["total_return_pct"] == 5.0
    
    def test_result_save_json(self):
        """Test saving result to JSON."""
        result = PipelineResult(
            pipeline_name="Test",
            start_time="2025-01-01T10:00:00",
            end_time="2025-01-01T10:05:00",
            duration_s=300.0
        )
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name
        
        try:
            result.save_json(temp_file)
            
            # Load and verify
            with open(temp_file) as f:
                loaded = json.load(f)
            
            assert loaded["pipeline_name"] == "Test"
        finally:
            Path(temp_file).unlink()
    
    def test_get_summary(self, sample_backtest_result):
        """Test result summary."""
        result = PipelineResult(
            pipeline_name="Test",
            start_time="2025-01-01T10:00:00",
            end_time="2025-01-01T10:05:00",
            duration_s=300.0,
            metrics={
                "total_return_pct": 10.0,
                "annualised_sharpe": 1.5,
                "max_drawdown_pct": -5.0,
                "n_bars": 100
            }
        )
        
        summary = result.get_summary()
        
        assert summary["pipeline_name"] == "Test"
        assert summary["duration_s"] == 300.0
        assert "metrics" in summary


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegrations:
    """Test integration modules."""
    
    def test_freqtrade_config_conversion(self, simple_config):
        """Test converting to freqtrade config."""
        ft_config = FreqtradeIntegration.config_from_pipeline(simple_config)
        
        assert "exchange" in ft_config
        assert ft_config["exchange"]["name"] == "portfoliobench"
        assert "pair_whitelist" in ft_config["exchange"]
    
    def test_pipeline_comparator(self):
        """Test pipeline comparison."""
        results = [
            PipelineResult(
                pipeline_name="Strategy A",
                start_time="2025-01-01T10:00:00",
                end_time="2025-01-01T10:05:00",
                duration_s=300.0,
                metrics={
                    "total_return_pct": 10.0,
                    "annualised_sharpe": 1.5,
                    "max_drawdown_pct": -5.0
                }
            ),
            PipelineResult(
                pipeline_name="Strategy B",
                start_time="2025-01-01T10:00:00",
                end_time="2025-01-01T10:15:00",
                duration_s=900.0,
                metrics={
                    "total_return_pct": 15.0,
                    "annualised_sharpe": 1.8,
                    "max_drawdown_pct": -3.0
                }
            )
        ]
        
        comparison = PipelineComparator.compare_results(results)
        
        assert len(comparison) == 2
        assert "Pipeline" in comparison.columns
        assert "Total Return (%)" in comparison.columns


# ============================================================================
# INTEGRATION: MAIN PIPELINE
# ============================================================================

class TestPortfolioPipeline:
    """Integration tests for main pipeline (mocked)."""
    
    @patch('pipeline.orchestrator.load_pair_data')
    @patch('pipeline.orchestrator.align_close_prices')
    def test_pipeline_initialization(self, mock_align, mock_load, simple_config):
        """Test pipeline initialization."""
        # Mock data loading
        mock_load.return_value = {
            "BTC/USDT": pd.DataFrame(),
            "ETH/USDT": pd.DataFrame()
        }
        
        pipeline = PortfolioPipeline(simple_config)
        
        assert pipeline.config.name == "Test Pipeline"
        assert pipeline.verbose is False


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

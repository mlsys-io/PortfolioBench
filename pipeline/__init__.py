"""
pipeline/
=========
End-to-end portfolio construction and backtesting pipeline system.

Modules:
--------
- orchestrator.py    : Main pipeline orchestrator and execution engine
- config.py          : Pipeline configuration management
- verification.py    : Validation and error checking
- results.py         : Pipeline result aggregation and reporting
- integrations.py    : Integration with freqtrade and external systems

Usage:
------
    from pipeline.orchestrator import Portfolio Pipeline
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig.from_yaml("pipelines/my_config.yaml")
    pipeline = PortfolioPipeline(cfg)
    results = pipeline.run()
"""

from __future__ import annotations

__all__ = [
    "PortfolioPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineVerification",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "PortfolioPipeline":
        from pipeline.orchestrator import PortfolioPipeline
        return PortfolioPipeline
    elif name == "PipelineConfig":
        from pipeline.config import PipelineConfig
        return PipelineConfig
    elif name == "PipelineResult":
        from pipeline.results import PipelineResult
        return PipelineResult
    elif name == "PipelineVerification":
        from pipeline.verification import PipelineVerification
        return PipelineVerification
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
pipeline/results.py
===================
Pipeline result aggregation and reporting.

Collects outputs from each pipeline stage and generates comprehensive reports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class StageOutput:
    """Output from a single pipeline stage."""
    name: str
    status: str  # "success", "failure", "skipped"
    duration_s: float
    data_summary: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    pipeline_name: str
    start_time: str
    end_time: str
    duration_s: float
    
    # Stage outputs
    stages: Dict[str, StageOutput] = field(default_factory=dict)
    
    # Key outputs
    pair_data: Optional[pd.DataFrame] = None
    enriched_data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    strategy_signals: Dict[str, pd.Series] = field(default_factory=dict)
    portfolio_weights: Optional[pd.DataFrame] = None
    backtest_result: Optional[pd.DataFrame] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation: Dict[str, Any] = field(default_factory=dict)
    
    # Settings
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamps are strings."""
        if not isinstance(self.start_time, str):
            self.start_time = str(self.start_time)
        if not isinstance(self.end_time, str):
            self.end_time = str(self.end_time)
    
    def add_stage_output(self, stage_name: str, output: StageOutput) -> None:
        """Record output from a pipeline stage."""
        self.stages[stage_name] = output
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary (excludes DataFrames/Series)."""
        return {
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_s": self.duration_s,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "metrics": self.metrics,
            "validation": self.validation,
            "config": self.config,
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save_json(self, filepath: str | Path) -> None:
        """Save result to JSON file."""
        Path(filepath).write_text(self.to_json())
    
    def save_weights_csv(self, filepath: str | Path) -> None:
        """Save portfolio weights to CSV."""
        if self.portfolio_weights is not None:
            self.portfolio_weights.to_csv(filepath)
        else:
            logger.warning("No portfolio weights to save")
    
    def save_backtest_csv(self, filepath: str | Path) -> None:
        """Save backtest results to CSV."""
        if self.backtest_result is not None:
            self.backtest_result.to_csv(filepath, index=False)
        else:
            logger.warning("No backtest results to save")
    
    def save_all(self, output_dir: str | Path) -> Dict[str, str]:
        """Save all outputs to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # JSON results
        json_file = output_dir / "pipeline_result.json"
        self.save_json(json_file)
        saved_files["result_json"] = str(json_file)
        
        # Portfolio weights
        if self.portfolio_weights is not None:
            weights_file = output_dir / "portfolio_weights.csv"
            self.save_weights_csv(weights_file)
            saved_files["weights_csv"] = str(weights_file)
        
        # Backtest results
        if self.backtest_result is not None:
            backtest_file = output_dir / "backtest_results.csv"
            self.save_backtest_csv(backtest_file)
            saved_files["backtest_csv"] = str(backtest_file)
        
        # HTML report
        report_file = output_dir / "report.html"
        report_file.write_text(self.to_html_report())
        saved_files["report_html"] = str(report_file)
        
        logger.info(f"Saved pipeline results to {output_dir}")
        return saved_files
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        # Stage summary
        stage_summary = {
            "total": len(self.stages),
            "succeeded": sum(1 for s in self.stages.values() if s.status == "success"),
            "failed": sum(1 for s in self.stages.values() if s.status == "failure"),
            "skipped": sum(1 for s in self.stages.values() if s.status == "skipped"),
        }
        
        # Metrics summary
        metrics_summary = {}
        if self.metrics:
            metrics_summary = {
                "total_return": self.metrics.get("total_return_pct"),
                "sharpe_ratio": self.metrics.get("annualised_sharpe"),
                "max_drawdown": self.metrics.get("max_drawdown_pct"),
                "n_bars": self.metrics.get("n_bars"),
            }
        
        # Validation summary
        validation_summary = {}
        if self.validation:
            validation_summary = {
                "total_checks": self.validation.get("total", 0),
                "passed": self.validation.get("passed", 0),
                "failed": self.validation.get("failed", 0),
                "pass_rate": self.validation.get("pass_rate", 0),
            }
        
        return {
            "pipeline_name": self.pipeline_name,
            "duration_s": self.duration_s,
            "stages": stage_summary,
            "metrics": metrics_summary,
            "validation": validation_summary,
        }
    
    def to_html_report(self) -> str:
        """Generate HTML report of pipeline results."""
        summary = self.get_summary()
        
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>PortfolioBench Pipeline Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }",
            ".header { background-color: #003366; color: white; padding: 20px; border-radius: 5px; }",
            ".section { background-color: white; margin: 15px 0; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".metric-grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; }",
            ".metric-box { background-color: #f9f9f9; padding: 10px; border-left: 4px solid #003366; }",
            ".metric-value { font-size: 22px; font-weight: bold; color: #003366; }",
            ".metric-label { font-size: 12px; color: #666; }",
            ".status-success { color: green; }",
            ".status-failure { color: red; }",
            ".status-skipped { color: orange; }",
            "table { width: 100%; border-collapse: collapse; }",
            "th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }",
            "th { background-color: #f0f0f0; font-weight: bold; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='header'>",
            f"<h1>PortfolioBench Pipeline Report</h1>",
            f"<p>Pipeline: {summary['pipeline_name']}</p>",
            f"<p>Duration: {summary['duration_s']:.2f}s</p>",
            "</div>",
        ]
        
        # Metrics section
        if summary['metrics']:
            html_parts.append("<div class='section'>")
            html_parts.append("<h2>Performance Metrics</h2>")
            html_parts.append("<div class='metric-grid'>")
            
            metrics = summary['metrics']
            if metrics.get('total_return') is not None:
                html_parts.append(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{metrics['total_return']:.2f}%</div>
                    <div class='metric-label'>Total Return</div>
                </div>
                """)
            
            if metrics.get('sharpe_ratio') is not None:
                html_parts.append(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{metrics['sharpe_ratio']:.4f}</div>
                    <div class='metric-label'>Sharpe Ratio</div>
                </div>
                """)
            
            if metrics.get('max_drawdown') is not None:
                html_parts.append(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{metrics['max_drawdown']:.2f}%</div>
                    <div class='metric-label'>Max Drawdown</div>
                </div>
                """)
            
            if metrics.get('n_bars') is not None:
                html_parts.append(f"""
                <div class='metric-box'>
                    <div class='metric-value'>{metrics['n_bars']}</div>
                    <div class='metric-label'>Bars Backtested</div>
                </div>
                """)
            
            html_parts.append("</div></div>")
        
        # Stages section
        html_parts.append("<div class='section'>")
        html_parts.append("<h2>Pipeline Stages</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Stage</th><th>Status</th><th>Duration (s)</th></tr>")
        
        for stage_name, stage in self.stages.items():
            status_class = f"status-{stage.status}"
            html_parts.append(f"""
            <tr>
                <td>{stage_name}</td>
                <td class='{status_class}'>{stage.status.upper()}</td>
                <td>{stage.duration_s:.3f}</td>
            </tr>
            """)
        
        html_parts.append("</table></div>")
        
        # Validation section
        if summary['validation']:
            val = summary['validation']
            html_parts.append("<div class='section'>")
            html_parts.append("<h2>Validation Results</h2>")
            html_parts.append("<div class='metric-grid'>")
            html_parts.append(f"""
            <div class='metric-box'>
                <div class='metric-value'>{val['total_checks']}</div>
                <div class='metric-label'>Total Checks</div>
            </div>
            """)
            html_parts.append(f"""
            <div class='metric-box'>
                <div class='metric-value' style='color: green;'>{val['passed']}</div>
                <div class='metric-label'>Passed</div>
            </div>
            """)
            html_parts.append(f"""
            <div class='metric-box'>
                <div class='metric-value' style='color: red;'>{val['failed']}</div>
                <div class='metric-label'>Failed</div>
            </div>
            """)
            html_parts.append(f"""
            <div class='metric-box'>
                <div class='metric-value'>{val['pass_rate']:.1f}%</div>
                <div class='metric-label'>Pass Rate</div>
            </div>
            """)
            html_parts.append("</div></div>")
        
        html_parts.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Pipeline:        {summary['pipeline_name']}")
        print(f"Duration:        {summary['duration_s']:.2f}s")
        
        stages = summary['stages']
        print(f"\nStages:          {stages['total']} total")
        print(f"  [+] Succeeded:   {stages['succeeded']}")
        print(f"  [-] Failed:      {stages['failed']}")
        print(f"  [~] Skipped:     {stages['skipped']}")
        
        if summary['metrics']:
            print(f"\nMetrics:")
            for key, value in summary['metrics'].items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"  {key:.<30} {value:>12.4f}")
                    else:
                        print(f"  {key:.<30} {value:>12}")
        
        if summary['validation']:
            val = summary['validation']
            print(f"\nValidation:      {val['total_checks']} checks")
            print(f"  [+] Passed:      {val['passed']}")
            print(f"  [-] Failed:      {val['failed']}")
            print(f"  Pass rate:     {val['pass_rate']:.1f}%")
        
        print("=" * 70 + "\n")

"""
pipeline/verification.py
========================
Validation and verification framework for portfolio pipelines.

Provides checks for data integrity, signal validity, weight constraints,
and risk metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details or {}
        }


class PipelineVerification:
    """Framework for validating pipeline components and outputs."""
    
    def __init__(self, strict: bool = False, verbose: bool = False):
        """
        Parameters
        ----------
        strict : if True, treat warnings as errors
        verbose : if True, log all validation details
        """
        self.strict = strict
        self.verbose = verbose
        self.results: List[ValidationResult] = []
    
    def validate_data_integrity(
        self,
        pair_data: Dict[str, pd.DataFrame],
        pairs: List[str],
    ) -> bool:
        """Validate loaded OHLCV data."""
        checks = []
        
        # Check 1: All pairs present
        missing_pairs = [p for p in pairs if p not in pair_data]
        if missing_pairs:
            result = ValidationResult(
                name="data_pairs_present",
                passed=False,
                message=f"Missing pairs: {missing_pairs}",
                details={"missing": missing_pairs}
            )
        else:
            result = ValidationResult(
                name="data_pairs_present",
                passed=True,
                message=f"All {len(pairs)} pairs loaded",
                details={"count": len(pairs)}
            )
        checks.append(result)
        
        # Check 2: Data completeness (no NaN in OHLCV)
        all_valid = True
        nan_counts = {}
        for pair, df in pair_data.items():
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            nan_count = df[ohlcv_cols].isna().sum().sum()
            if nan_count > 0:
                all_valid = False
                nan_counts[pair] = int(nan_count)
        
        if all_valid:
            result = ValidationResult(
                name="data_completeness",
                passed=True,
                message="No missing values in OHLCV data",
                details={"pairs_checked": len(pair_data)}
            )
        else:
            result = ValidationResult(
                name="data_completeness",
                passed=False,
                message=f"Found NaN values in {len(nan_counts)} pairs",
                details=nan_counts
            )
        checks.append(result)
        
        # Check 3: Date alignment
        all_dates = set()
        date_ranges = {}
        for pair, df in pair_data.items():
            dates = set(df["date"])
            date_ranges[pair] = (df["date"].min(), df["date"].max())
            if not all_dates:
                all_dates = dates
            else:
                all_dates = all_dates & dates
        
        if all_dates:
            result = ValidationResult(
                name="data_date_alignment",
                passed=True,
                message=f"Found {len(all_dates)} common dates across all pairs",
                details={"common_dates": len(all_dates)}
            )
        else:
            result = ValidationResult(
                name="data_date_alignment",
                passed=False,
                message="No overlapping dates found",
                details={"date_ranges": {
                    k: (str(v[0]), str(v[1])) for k, v in date_ranges.items()
                }}
            )
        checks.append(result)
        
        # Check 4: OHLC relationships (high >= low, etc.)
        ohlc_valid = True
        violations = {}
        for pair, df in pair_data.items():
            violations[pair] = 0
            # High >= Low
            violations[pair] += (df["high"] < df["low"]).sum()
            # Close between High and Low
            violations[pair] += (
                (df["close"] > df["high"]) | (df["close"] < df["low"])
            ).sum()
        
        ohlc_valid = all(v == 0 for v in violations.values())
        if ohlc_valid:
            result = ValidationResult(
                name="data_ohlc_relationships",
                passed=True,
                message="OHLC relationships valid (H>=L, Close between H and L)",
                details={"pairs_checked": len(pair_data)}
            )
        else:
            result = ValidationResult(
                name="data_ohlc_relationships",
                passed=False,
                message=f"Found OHLC violations in {len([v for v in violations.values() if v > 0])} pairs",
                details={k: int(v) for k, v in violations.items() if v > 0}
            )
        checks.append(result)
        
        # Check 5: Volume > 0
        volume_valid = True
        zero_volume_pairs = {}
        for pair, df in pair_data.items():
            zero_count = (df["volume"] <= 0).sum()
            if zero_count > 0:
                volume_valid = False
                zero_volume_pairs[pair] = int(zero_count)
        
        if volume_valid:
            result = ValidationResult(
                name="data_volume_positive",
                passed=True,
                message="All volume values > 0",
                details={"pairs_checked": len(pair_data)}
            )
        else:
            result = ValidationResult(
                name="data_volume_positive",
                passed=False,
                message=f"Found zero/negative volume in {len(zero_volume_pairs)} pairs",
                details=zero_volume_pairs
            )
        checks.append(result)
        
        self.results.extend(checks)
        return all(c.passed for c in checks)
    
    def validate_alpha_signals(
        self,
        enriched_data: Dict[str, pd.DataFrame],
        expected_columns: List[str],
    ) -> bool:
        """Validate alpha factor outputs."""
        checks = []
        
        # Check 1: Required columns present
        all_cols_present = True
        missing_cols_by_pair = {}
        for pair, df in enriched_data.items():
            missing = [c for c in expected_columns if c not in df.columns]
            if missing:
                all_cols_present = False
                missing_cols_by_pair[pair] = missing
        
        if all_cols_present:
            result = ValidationResult(
                name="alpha_columns_present",
                passed=True,
                message=f"All expected alpha columns present",
                details={"expected_columns": expected_columns}
            )
        else:
            result = ValidationResult(
                name="alpha_columns_present",
                passed=False,
                message=f"Missing columns in {len(missing_cols_by_pair)} pairs",
                details=missing_cols_by_pair
            )
        checks.append(result)
        
        # Check 2: No NaN in critical columns
        no_nan = True
        nan_summary = {}
        for pair, df in enriched_data.items():
            for col in expected_columns:
                if col in df.columns:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        no_nan = False
                        if pair not in nan_summary:
                            nan_summary[pair] = {}
                        nan_summary[pair][col] = int(nan_count)
        
        if no_nan:
            result = ValidationResult(
                name="alpha_columns_no_nan",
                passed=True,
                message="No NaN values in alpha columns",
                details={"pairs_checked": len(enriched_data)}
            )
        else:
            result = ValidationResult(
                name="alpha_columns_no_nan",
                passed=False,
                message=f"Found NaN values in alpha columns",
                details=nan_summary
            )
        checks.append(result)
        
        # Check 3: Signal value ranges
        signal_valid = True
        signal_ranges = {}
        for pair, df in enriched_data.items():
            signal_ranges[pair] = {}
            # Check binary signals
            for col in ["enter_long", "exit_long", "rsi_oversold", "rsi_overbought"]:
                if col in df.columns:
                    unique = df[col].unique()
                    if not all(v in [0, 1, np.nan] for v in unique):
                        signal_valid = False
                    signal_ranges[pair][col] = list(map(float, unique[~pd.isna(unique)]))
        
        if signal_valid:
            result = ValidationResult(
                name="alpha_signal_ranges",
                passed=True,
                message="Signal values in valid ranges",
                details={"pairs_checked": len(enriched_data)}
            )
        else:
            result = ValidationResult(
                name="alpha_signal_ranges",
                passed=False,
                message="Invalid signal values detected",
                details=signal_ranges
            )
        checks.append(result)
        
        self.results.extend(checks)
        return all(c.passed for c in checks)
    
    def validate_strategy_signals(
        self,
        strategy_signals: Dict[str, pd.Series],
    ) -> bool:
        """Validate trading strategy signals."""
        checks = []
        
        # Check 1: All signals binary
        all_binary = True
        non_binary = {}
        for pair, signals in strategy_signals.items():
            unique = signals.unique()
            if not all(v in [0, 1] for v in unique if not pd.isna(v)):
                all_binary = False
                non_binary[pair] = list(unique)
        
        if all_binary:
            result = ValidationResult(
                name="strategy_signals_binary",
                passed=True,
                message="All strategy signals are binary (0/1)",
                details={"pairs_checked": len(strategy_signals)}
            )
        else:
            result = ValidationResult(
                name="strategy_signals_binary",
                passed=False,
                message="Non-binary signals found",
                details=non_binary
            )
        checks.append(result)
        
        # Check 2: Signal statistics
        signal_stats = {}
        for pair, signals in strategy_signals.items():
            signal_count = (signals == 1).sum()
            signal_pct = (signal_count / len(signals) * 100) if len(signals) > 0 else 0
            signal_stats[pair] = {
                "count": int(signal_count),
                "pct": round(signal_pct, 2)
            }
        
        result = ValidationResult(
            name="strategy_signal_density",
            passed=True,
            message="Strategy signal statistics computed",
            details=signal_stats
        )
        checks.append(result)
        
        self.results.extend(checks)
        return all(c.passed for c in checks)
    
    def validate_portfolio_weights(
        self,
        weights: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> bool:
        """Validate portfolio weight constraints."""
        checks = []
        
        # Check 1: Weights sum to 1.0
        row_sums = weights.sum(axis=1)
        sum_valid = np.allclose(row_sums, 1.0, atol=tolerance)
        
        if sum_valid:
            result = ValidationResult(
                name="weights_sum_to_one",
                passed=True,
                message=f"All weight rows sum to 1.0 (±{tolerance})",
                details={
                    "min_sum": float(row_sums.min()),
                    "max_sum": float(row_sums.max()),
                    "mean_sum": float(row_sums.mean())
                }
            )
        else:
            bad_rows = row_sums[~np.isclose(row_sums, 1.0, atol=tolerance)]
            result = ValidationResult(
                name="weights_sum_to_one",
                passed=False,
                message=f"Found {len(bad_rows)} rows with invalid weight sums",
                details={
                    "min_sum": float(row_sums.min()),
                    "max_sum": float(row_sums.max()),
                    "bad_row_count": len(bad_rows)
                }
            )
        checks.append(result)
        
        # Check 2: Weights in [0, 1]
        in_bounds = (weights >= 0).all().all() and (weights <= 1).all().all()
        
        if in_bounds:
            result = ValidationResult(
                name="weights_in_bounds",
                passed=True,
                message="All weights in [0, 1]",
                details={
                    "min_weight": float(weights.min().min()),
                    "max_weight": float(weights.max().max())
                }
            )
        else:
            neg_weights = (weights < 0).sum().sum()
            over_weights = (weights > 1).sum().sum()
            result = ValidationResult(
                name="weights_in_bounds",
                passed=False,
                message=f"Found weights outside [0, 1]: {neg_weights} negative, {over_weights} > 1",
                details={
                    "min_weight": float(weights.min().min()),
                    "max_weight": float(weights.max().max())
                }
            )
        checks.append(result)
        
        # Check 3: Weight diversity (not overly concentrated)
        # Average Herfindahl index per row
        hhi = (weights ** 2).sum(axis=1).mean()
        n_assets = weights.shape[1]
        hhi_threshold = 1.0 / n_assets  # Equal weight threshold
        
        result = ValidationResult(
            name="weight_concentration",
            passed=True,
            message=f"Portfolio concentration metrics computed",
            details={
                "mean_hhi": round(hhi, 4),
                "n_assets": n_assets,
                "equal_weight_hhi": round(hhi_threshold, 4),
                "concentration_ratio": round(hhi / hhi_threshold, 2)
            }
        )
        checks.append(result)
        
        self.results.extend(checks)
        return all(c.passed for c in checks)
    
    def validate_backtest_results(
        self,
        backtest_result: pd.DataFrame,
    ) -> bool:
        """Validate backtest output."""
        checks = []
        
        # Check 1: Required columns
        required = ["date", "portfolio_value", "daily_return"]
        has_cols = all(c in backtest_result.columns for c in required)
        
        result = ValidationResult(
            name="backtest_columns",
            passed=has_cols,
            message="Backtest result has required columns" if has_cols else "Missing required columns",
            details={"required": required, "present": list(backtest_result.columns)}
        )
        checks.append(result)
        
        # Check 2: Portfolio value monotonicity
        pv = backtest_result["portfolio_value"]
        # Value can go up or down, but shouldn't be zero
        no_zero = (pv > 0).all()
        
        result = ValidationResult(
            name="backtest_portfolio_value_valid",
            passed=no_zero,
            message="Portfolio values all positive" if no_zero else "Found zero or negative portfolio values",
            details={
                "min_value": float(pv.min()),
                "max_value": float(pv.max()),
                "mean_value": float(pv.mean())
            }
        )
        checks.append(result)
        
        # Check 3: Return statistics
        returns = backtest_result["daily_return"]
        total_return = (pv.iloc[-1] / pv.iloc[0]) - 1
        sharpe_approx = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        result = ValidationResult(
            name="backtest_return_statistics",
            passed=True,
            message="Return statistics computed",
            details={
                "total_return": round(total_return, 4),
                "mean_daily_return": round(returns.mean(), 6),
                "std_daily_return": round(returns.std(), 6),
                "approx_sharpe": round(sharpe_approx, 4),
                "min_return": round(returns.min(), 4),
                "max_return": round(returns.max(), 4)
            }
        )
        checks.append(result)
        
        self.results.extend(checks)
        return all(c.passed for c in checks)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "results": []}
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / len(self.results) * 100, 1),
            "results": [r.to_dict() for r in self.results]
        }
    
    def report(self) -> str:
        """Generate human-readable validation report."""
        summary = self.get_summary()
        
        if summary["total"] == 0:
            return "No validation results"
        
        lines = [
            "=" * 70,
            "PIPELINE VALIDATION REPORT",
            "=" * 70,
            f"Total checks: {summary['total']}",
            f"Passed: {summary['passed']}",
            f"Failed: {summary['failed']}",
            f"Pass rate: {summary['pass_rate']}%",
            "",
            "-" * 70,
        ]
        
        for result in self.results:
            status = "[+] PASS" if result.passed else "[-] FAIL"
            lines.append(f"{status}  {result.name}")
            lines.append(f"     {result.message}")
            if result.details:
                details_str = str(result.details)[:100]
                lines.append(f"     Details: {details_str}...")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)

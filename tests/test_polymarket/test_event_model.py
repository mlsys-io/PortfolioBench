"""Tests for the direct event-probability model pipeline.

All tests use fully synthetic BTC data so no real data files are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from polymarket.event_dataset import build_training_samples
from polymarket.event_features import (
    ALL_FEATURE_COLS,
    BTC_FEATURE_COLS,
    CONTRACT_FEATURE_COLS,
    ONCHAIN_FEATURE_COLS,
    add_btc_features,
    add_contract_features,
    build_feature_matrix,
)
from polymarket.event_model import (
    load_model,
    predict_contract_probs,
    save_model,
    train,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_btc_df(n_rows: int = 2500, seed: int = 0) -> pd.DataFrame:
    """Synthetic BTC OHLCV + on-chain DataFrame (~104 days at default size)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01", tz="UTC")
    dt = pd.date_range(start=start, periods=n_rows, freq="h", tz="UTC")
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.003, n_rows)))
    df = pd.DataFrame({
        "dt": dt,
        "Open":   close * (1 + rng.normal(0, 0.001, n_rows)),
        "High":   close * (1 + np.abs(rng.normal(0, 0.003, n_rows))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.003, n_rows))),
        "Close":  close,
        "Volume": rng.uniform(100, 1000, n_rows),
        "mvrv":         rng.uniform(1.0, 3.0, n_rows),
        "hash-rate":    rng.uniform(200, 400, n_rows),
        "difficulty":   rng.uniform(30e12, 60e12, n_rows),
    })
    return df


@pytest.fixture
def btc_df() -> pd.DataFrame:
    return _make_btc_df(n_rows=2500)


@pytest.fixture
def settlement_T(btc_df) -> pd.Timestamp:
    # Use a timestamp 1000 hours into the series so rolling features are warm
    return btc_df["dt"].iloc[1000]


@pytest.fixture
def strike(btc_df, settlement_T) -> float:
    # ATM: strike = BTC price 7 days before T
    idx = btc_df[btc_df["dt"] == settlement_T].index[0]
    ref_idx = max(0, idx - 168)
    return float(btc_df["Close"].iloc[ref_idx])


# ---------------------------------------------------------------------------
# event_features — add_btc_features
# ---------------------------------------------------------------------------

class TestAddBtcFeatures:

    def test_returns_copy(self, btc_df):
        result = add_btc_features(btc_df)
        assert result is not btc_df

    def test_btc_feature_columns_present(self, btc_df):
        result = add_btc_features(btc_df)
        for col in BTC_FEATURE_COLS:
            assert col in result.columns, f"Missing BTC feature: {col}"

    def test_onchain_lag_columns_present(self, btc_df):
        result = add_btc_features(btc_df)
        for col in ONCHAIN_FEATURE_COLS:
            assert col in result.columns, f"Missing on-chain feature: {col}"

    def test_log_ret_1h_is_causal(self, btc_df):
        """log_ret_1h at row i must only depend on rows 0..i."""
        result = add_btc_features(btc_df)
        # At row i, log_ret_1h = log(Close[i] / Close[i-1])
        # Verify using rows 10..15 where rolling windows are warm
        for i in range(10, 15):
            expected = np.log(btc_df["Close"].iloc[i] / btc_df["Close"].iloc[i - 1])
            assert abs(result["log_ret_1h"].iloc[i] - expected) < 1e-10

    def test_mvrv_lag24_is_lagged(self, btc_df):
        result = add_btc_features(btc_df)
        # At row i, mvrv_lag24 should equal mvrv at row i-24
        for i in [30, 50, 100]:
            expected = btc_df["mvrv"].iloc[i - 24]
            got = result["mvrv_lag24"].iloc[i]
            assert abs(got - expected) < 1e-10, f"mvrv_lag24 mismatch at row {i}"

    def test_missing_onchain_produces_nan(self):
        """DataFrames without on-chain columns should produce NaN lag features."""
        df = _make_btc_df(200)
        df = df[["dt", "Open", "High", "Low", "Close", "Volume"]]
        result = add_btc_features(df)
        assert result["mvrv_lag24"].isna().all()

    def test_no_future_leakage_in_rolling(self, btc_df):
        """Truncating the df at row N must not change values at rows < N."""
        full = add_btc_features(btc_df)
        partial = add_btc_features(btc_df.iloc[:300])
        # Both should agree on rows 200..299
        for col in BTC_FEATURE_COLS:
            diff = (full[col].iloc[200:300].values - partial[col].iloc[200:300].values)
            assert np.allclose(diff, 0, equal_nan=True), (
                f"Column {col} differs between full and partial df at rows 200-299"
            )


# ---------------------------------------------------------------------------
# event_features — add_contract_features
# ---------------------------------------------------------------------------

class TestAddContractFeatures:

    def test_contract_columns_present(self, btc_df, strike, settlement_T):
        df = add_btc_features(btc_df)
        result = add_contract_features(df, strike, settlement_T)
        for col in CONTRACT_FEATURE_COLS + ["h_remaining"]:
            assert col in result.columns

    def test_log_moneyness_sign(self, btc_df, settlement_T):
        df = add_btc_features(btc_df)
        K_high = btc_df["Close"].max() * 2   # always OTM
        K_low  = btc_df["Close"].min() / 2   # always ITM
        result_high = add_contract_features(df, K_high, settlement_T)
        result_low  = add_contract_features(df, K_low,  settlement_T)
        assert (result_high["log_moneyness"] < 0).all()
        assert (result_low["log_moneyness"] > 0).all()

    def test_h_remaining_decreases_toward_T(self, btc_df, strike, settlement_T):
        df = add_btc_features(btc_df)
        result = add_contract_features(df, strike, settlement_T)
        # Filter to window before T
        window = result[result["dt"] < settlement_T]
        if len(window) > 1:
            diffs = window["h_remaining"].diff().dropna()
            assert (diffs <= 0).all(), "h_remaining should be monotonically decreasing"

    def test_h_remaining_floored_at_half_hour(self, btc_df, strike, settlement_T):
        df = add_btc_features(btc_df)
        result = add_contract_features(df, strike, settlement_T)
        assert (result["h_remaining"] >= 0.5).all()

    def test_bs_prob_in_range(self, btc_df, strike, settlement_T):
        df = add_btc_features(btc_df)
        result = add_contract_features(df, strike, settlement_T)
        valid = result["bs_prob"].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_bs_prob_atm_near_half(self, btc_df, settlement_T):
        """ATM contract deep in the window should have bs_prob ≈ 0.5."""
        df = add_btc_features(btc_df)
        # Use the BTC price right at the start of the window as the strike
        window_start_idx = btc_df[btc_df["dt"] == settlement_T].index[0] - 168
        K_atm = float(btc_df["Close"].iloc[window_start_idx])
        result = add_contract_features(df, K_atm, settlement_T)
        # At the start of the window (h_remaining ≈ 168), ATM bs_prob should be
        # somewhere near 0.5 (within ±0.2 for a 7-day BTC window)
        early_row = result[result["dt"] == btc_df["dt"].iloc[window_start_idx]]
        if not early_row.empty:
            p = early_row["bs_prob"].iloc[0]
            assert 0.2 < p < 0.8, f"ATM bs_prob={p:.3f} is too far from 0.5"


# ---------------------------------------------------------------------------
# event_features — build_feature_matrix
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:

    def test_output_has_all_feature_cols(self, btc_df, strike, settlement_T):
        result = build_feature_matrix(btc_df, strike, settlement_T)
        for col in ALL_FEATURE_COLS:
            assert col in result.columns

    def test_no_nan_in_feature_cols(self, btc_df, strike, settlement_T):
        result = build_feature_matrix(btc_df, strike, settlement_T)
        X = result[ALL_FEATURE_COLS]
        assert not X.isnull().any().any(), "Feature matrix contains NaN values"

    def test_dt_column_preserved(self, btc_df, strike, settlement_T):
        result = build_feature_matrix(btc_df, strike, settlement_T)
        assert "dt" in result.columns

    def test_all_rows_before_T(self, btc_df, strike, settlement_T):
        result = build_feature_matrix(btc_df, strike, settlement_T)
        # build_feature_matrix doesn't filter to window; caller does.
        # Just check that h_remaining >= 0.5 for all rows.
        assert (result["h_remaining"] >= 0.5).all()


# ---------------------------------------------------------------------------
# event_dataset — build_training_samples
# ---------------------------------------------------------------------------

class TestBuildTrainingSamples:

    def test_basic_shape(self, btc_df):
        # Use a narrow date range to keep the test fast
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[0.95, 1.00, 1.05],
        )
        assert len(samples) > 0
        for col in ALL_FEATURE_COLS:
            assert col in samples.columns
        assert "label" in samples.columns
        assert "K" in samples.columns
        assert "T" in samples.columns

    def test_label_binary(self, btc_df):
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[1.00],
        )
        assert set(samples["label"].unique()).issubset({0, 1})

    def test_no_nan_features(self, btc_df):
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[0.95, 1.00, 1.05],
        )
        assert not samples[ALL_FEATURE_COLS].isnull().any().any()

    def test_T_column_tz_aware(self, btc_df):
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[1.00],
        )
        assert samples["T"].dt.tz is not None

    def test_h_remaining_positive(self, btc_df):
        """All samples must be strictly before their settlement time T."""
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[1.00],
        )
        assert (samples["h_remaining"] > 0).all()

    def test_label_consistency(self, btc_df):
        """All rows with the same (K, T) must have the same label."""
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[1.00],
        )
        for (K, T), group in samples.groupby(["K", "T"], observed=True):
            assert group["label"].nunique() == 1, (
                f"Label is not constant within event (K={K}, T={T})"
            )

    def test_temporal_ordering_preserved(self, btc_df):
        """Samples for a given (K, T) must be ordered by ascending dt."""
        samples = build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[1.00],
        )
        for (K, T), group in samples.groupby(["K", "T"], observed=True):
            dts = group["dt"].values
            assert (dts[1:] > dts[:-1]).all(), (
                f"Timestamps not monotonically increasing for K={K}, T={T}"
            )


# ---------------------------------------------------------------------------
# event_model — train / predict / save / load
# ---------------------------------------------------------------------------

class TestEventModel:

    @pytest.fixture
    def small_samples(self, btc_df):
        """Small but valid training samples for model tests."""
        return build_training_samples(
            btc_df,
            start_date="2023-02-01",
            end_date="2023-04-01",
            window_days=7,
            relative_strikes=[0.95, 1.00, 1.05],
        )

    def test_train_returns_package_keys(self, small_samples):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        for key in ("model", "feature_cols", "val_cutoff", "metrics", "model_type"):
            assert key in pkg

    def test_feature_cols_match_all_feature_cols(self, small_samples):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        assert pkg["feature_cols"] == ALL_FEATURE_COLS

    def test_metrics_contain_splits(self, small_samples):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        assert "train" in pkg["metrics"]
        assert "held" in pkg["metrics"]

    def test_train_val_split_error_on_empty_train(self, small_samples):
        with pytest.raises(ValueError, match="empty"):
            train(small_samples, val_cutoff="2020-01-01")  # before all data

    def test_save_load_roundtrip(self, small_samples, tmp_path):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        model_path = tmp_path / "event_model.pkl"
        save_model(pkg, model_path)
        loaded = load_model(model_path)
        assert loaded["model_type"] == "logistic"
        assert loaded["feature_cols"] == ALL_FEATURE_COLS

    def test_load_model_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")

    def test_predict_contract_probs_output_shape(self, btc_df, small_samples, settlement_T, strike):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        probs = predict_contract_probs(btc_df, strike, settlement_T, pkg)
        assert "dt_utc" in probs.columns
        assert "fair_value" in probs.columns
        assert len(probs) > 0

    def test_predict_contract_probs_in_range(self, btc_df, small_samples, settlement_T, strike):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        probs = predict_contract_probs(btc_df, strike, settlement_T, pkg)
        assert (probs["fair_value"] >= 0.0).all()
        assert (probs["fair_value"] <= 1.0).all()

    def test_predict_only_before_T(self, btc_df, small_samples, settlement_T, strike):
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        probs = predict_contract_probs(btc_df, strike, settlement_T, pkg)
        assert (probs["dt_utc"] < settlement_T).all()

    def test_high_strike_low_fair_value(self, btc_df, small_samples, settlement_T):
        """Very high strike should produce low fair value (unlikely to win)."""
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        K_very_high = btc_df["Close"].max() * 5
        probs = predict_contract_probs(btc_df, K_very_high, settlement_T, pkg)
        assert probs["fair_value"].mean() < 0.3, (
            "Very high strike should yield low probability"
        )

    def test_low_strike_high_fair_value(self, btc_df, small_samples, settlement_T):
        """Very low strike should produce high fair value (very likely to win)."""
        pkg = train(small_samples, val_cutoff="2023-03-01", model_type="logistic")
        K_very_low = btc_df["Close"].min() / 5
        probs = predict_contract_probs(btc_df, K_very_low, settlement_T, pkg)
        assert probs["fair_value"].mean() > 0.7, (
            "Very low strike should yield high probability"
        )

"""Kalshi Economic Events — Implied Mean Surprise Direction Baseline.

Recovers implied means from Kalshi threshold-based event markets
(CPI, Core CPI, FED level/decision, Payrolls, Unemployment, GDP)
via the stg_infra events submodule and trains a logistic regression
baseline to predict surprise direction:

  label=1  → actual print > implied mean  (upside surprise)
  label=0  → actual print < implied mean  (downside surprise)

Features come directly from the PDF-derived statistics returned by the
stg_infra event series functions: implied_mean, implied_std, implied_skew,
implied_kurtosis, implied_entropy, implied_median, n_submarkets.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# stg_infra uses two import roots:
#   - PortfolioBench/stg          → makes `stg_infra.*` findable (namespace pkg)
#   - PortfolioBench/stg/stg_infra → makes `stg.*` findable (installed pkg root)
_STG_ROOT = Path(__file__).parent.parent / "stg"
sys.path.insert(0, str(_STG_ROOT))
sys.path.insert(0, str(_STG_ROOT / "stg_infra"))

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from stg.events.cpi import compute_cpi_mom_series, compute_cpi_yoy_series
from stg.events.core_cpi import compute_core_cpi_mom_series, compute_core_cpi_yoy_series
from stg.events.fed import compute_fed_level_series, compute_fed_decision_series
from stg.events.payrolls import compute_payrolls_series
from stg.events.unemployment import compute_unemployment_series
from stg.events.gdp import compute_gdp_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR    = Path("user_data/data/kalshi/kalshi")
TRAIN_RATIO = 0.7

SERIES_ORDER = [
    "cpi_mom", "cpi_yoy",
    "core_cpi_mom", "core_cpi_yoy",
    "fed_level", "decision_bps",
    "payrolls",
    "unemployment",
    "gdp",
]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
log.info("Loading markets...")
markets = pl.read_parquet(str(DATA_DIR / "markets/*.parquet"))

log.info("Loading trades...")
trades = pl.read_parquet(str(DATA_DIR / "trades/*.parquet"))

log.info("Markets: %d rows | Trades: %d rows", len(markets), len(trades))

# ---------------------------------------------------------------------------
# 2. Compute implied mean series for every supported event type
# ---------------------------------------------------------------------------
log.info("Computing implied mean series...")

_series_fns = [
    compute_cpi_mom_series,
    compute_cpi_yoy_series,
    compute_core_cpi_mom_series,
    compute_core_cpi_yoy_series,
    compute_fed_level_series,
    compute_fed_decision_series,
    compute_payrolls_series,
    compute_unemployment_series,
    compute_gdp_series,
]

frames = []
for fn in _series_fns:
    df = fn(markets, trades)
    if not df.is_empty():
        frames.append(df)
        log.info("  %-25s  %d rows  %d events",
                 fn.__name__, len(df), df["event_ticker"].n_unique())

if not frames:
    sys.exit("No implied mean data found.")

implied = pl.concat(frames, how="diagonal").sort(["event_ticker", "date"])
log.info("Total implied-mean rows: %d across %d events",
         len(implied), implied["event_ticker"].n_unique())

# ---------------------------------------------------------------------------
# 3. Label each observation
# ---------------------------------------------------------------------------
# Keep only rows where resolved_value is known and implied_mean is not null
# and there is a clear surprise (skip exact ties)
labeled = (
    implied
    .filter(
        pl.col("resolved_value").is_not_null()
        & pl.col("implied_mean").is_not_null()
        & (pl.col("resolved_value") != pl.col("implied_mean"))
    )
    .with_columns(
        pl.when(pl.col("resolved_value") > pl.col("implied_mean"))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("label")
    )
)

log.info("Labeled observations: %d (up=%d, down=%d)",
         len(labeled),
         (labeled["label"] == 1).sum(),
         (labeled["label"] == 0).sum())

if labeled.is_empty():
    sys.exit("No labeled observations after filtering.")

# ---------------------------------------------------------------------------
# 4. Chronological train / test split  (split on date, not shuffle)
# ---------------------------------------------------------------------------
all_dates = labeled["date"].sort()
split_idx  = int(len(all_dates) * TRAIN_RATIO)
split_date = all_dates[split_idx]

train_df = labeled.filter(pl.col("date") < split_date)
test_df  = labeled.filter(pl.col("date") >= split_date)

log.info("Train: %d obs (%s → %s)", len(train_df),
         train_df["date"].min(), train_df["date"].max())
log.info("Test:  %d obs (%s → %s)", len(test_df),
         test_df["date"].min(), test_df["date"].max())

# ---------------------------------------------------------------------------
# 5. Assemble feature matrix
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "implied_mean",
    "implied_std",
    "implied_skew",
    "implied_kurtosis",
    "implied_entropy",
    "implied_median",
    "n_submarkets",
]


def to_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float64)
    y = df["label"].to_numpy().astype(np.int64)
    # Drop rows with any NaN feature (lag windows produce NaNs at start of series)
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


X_train, y_train = to_xy(train_df)
X_test,  y_test  = to_xy(test_df)

log.info("Train samples after NaN drop: %d (up=%d, down=%d)",
         len(y_train), (y_train == 1).sum(), (y_train == 0).sum())
log.info("Test  samples after NaN drop: %d (up=%d, down=%d)",
         len(y_test), (y_test == 1).sum(), (y_test == 0).sum())

if len(np.unique(y_train)) < 2:
    sys.exit("Training set has only one class — cannot train classifier.")

# ---------------------------------------------------------------------------
# 6. Train logistic regression baseline
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

# ---------------------------------------------------------------------------
# 7. Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Implied Mean Surprise Direction Baseline (Logistic Regression)")
print("=" * 60)
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print()
print(classification_report(y_test, y_pred, target_names=["downside", "upside"]))

coefs = model.coef_[0]
print("Feature coefficients (sorted by magnitude):")
for name, coef in sorted(zip(FEATURE_COLS, coefs), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:20s}  {coef:+.4f}")

# Per-series breakdown
print("\nPer-series accuracy:")
for stype in SERIES_ORDER:
    sub = test_df.filter(pl.col("series_type") == stype)
    if sub.is_empty():
        continue
    Xs, ys = to_xy(sub)
    if len(Xs) == 0:
        continue
    Xs_s = scaler.transform(Xs)
    preds = model.predict(Xs_s)
    acc = accuracy_score(ys, preds)
    print(f"  {stype:20s}  n={len(ys):4d}  acc={acc:.4f}")

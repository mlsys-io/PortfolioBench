# Polymarket Event-Probability Model — Training Guide

This document is the technical reference for training, evaluating, and extending the
direct event-probability model. See [README.md](README.md) for the overview and quick
start.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Input Data Requirements](#2-input-data-requirements)
3. [Feature Engineering](#3-feature-engineering)
4. [Label Construction](#4-label-construction)
5. [Temporal Split and Leakage Prevention](#5-temporal-split-and-leakage-prevention)
6. [Model Training and Calibration](#6-model-training-and-calibration)
7. [Inference on New Contracts](#7-inference-on-new-contracts)
8. [Temporal Usage Rules](#8-temporal-usage-rules)
9. [Training Artefacts](#9-training-artefacts)
10. [Retraining Guidance](#10-retraining-guidance)
11. [Extending Predictions](#11-extending-predictions)

---

## 1. Architecture Overview

```
                ┌─────────────────────────────────────────────┐
                │  Step 0: OHLCV feather files per contract    │
                │                                              │
                │  Option A (default — synthetic):             │
                │    build_all_feathers()    (data_builder.py) │
                │    • Log-normal price simulation             │
                │                                              │
                │  Option B (--use-real-data):                 │
                │    build_all_feathers_from_parquet()         │
                │                    (real_data_builder.py)    │
                │    • Real Polymarket YES-side prices         │
                │    • Forward-fill gaps ≤ 6h                  │
                │    • Writes real_contracts.jsonl             │
                └─────────────────────────────────────────────┘

data_1h.csv                     (BTC hourly OHLCV, 2018–present)
      │
      ▼
build_training_samples()        (event_dataset.py)
  • Generate weekly settlement times (Mondays, 17:00 UTC)
  • Apply 7 relative strike multipliers per event
  • Compute 15 features at every hourly bar before settlement
  • Assign binary label: BTC_Open_at_T > K
      │
      ▼
event_model_training.parquet    (363K rows × 18 columns)
      │
      ▼
train()                         (event_model.py)
  • Temporal split: T < 2024-01-01 → train
  • Calibration fold: first half of validation window
  • Held-out fold: second half of validation window
  • Fit: StandardScaler → LogisticRegression
  • Calibrate: IsotonicRegression on calibration fold scores
      │
      ▼
event_model.pkl                 (model package dict)
      │
      ▼
predict_contract_probs()        (event_model.py)
  • For each real Polymarket contract (K, T):
    • Compute features at every hourly bar before T
    • Return calibrated P(BTC_T > K) ∈ (0, 1)
      │
      ▼
{pair}-event_probs.csv          (one file per contract)
      │
      ▼
DualModelPolymarketPortfolio    (user_data/strategies/)
  • EventProbAlpha reads fair_value column
  • Computes edge = fair_value − market_price
  • Fractional Kelly sizing with per-expiry cap
```

---

## 2. Input Data Requirements

### BTC hourly OHLCV CSV

**Default path:** `mycode/data/data_1h.csv`

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `Timestamp` | int | Unix epoch **seconds** |
| `Open` | float | Hourly open price (USD) |
| `High` | float | Hourly high price (USD) |
| `Low` | float | Hourly low price (USD) |
| `Close` | float | Hourly close price (USD) |
| `Volume` | float | Hourly traded volume |

**Optional on-chain columns** (improve model quality; omit if unavailable):

| Column | Description |
|---|---|
| `mvrv` | Market-value-to-realised-value ratio |
| `hash-rate` | Network hash rate |
| `difficulty` | Mining difficulty |

On-chain columns are shifted 24 hours backward to avoid lookahead (see
[Section 3](#3-feature-engineering)).

**Coverage requirement:**

- For training: at least from `start_date` minus `window_days` (typically 7 days) to
  capture the first event's feature window.
- For inference: at least 7 days of history before the first bar you want predictions
  for.

### Contract metadata JSONL

**Default path:** `user_data/data/polymarket_contracts/jan20.jsonl`

One JSON object per line. Required fields per contract:

```json
{
  "question":      "Will the price of Bitcoin be above $90,000 on January 20?",
  "endDate":       "2026-01-20T17:00:00Z",
  "startDate":     "2026-01-13T17:03:21Z",
  "outcomePrices": "[\"1\", \"0\"]",
  "id":            "1176162",
  "slug":          "bitcoin-above-90k-on-january-20"
}
```

**Field notes:**
- `question`: used for strike and direction extraction. Supported patterns:
  - `"above $X,XXX"` / `"above $XXK"` — YES if BTC > K at expiry
  - `"reach/hit/exceed/surpass $X,XXX"` — treated as "above"
  - `"below $X,XXX"` / `"less than $X,XXX"` / `"dips to $X,XXX"` — YES if BTC < K
  - K-suffix supported: `$88K` is parsed as `$88,000`
  - Lines whose `question` cannot be parsed are skipped with a warning when
    `skip_unparseable=True`; they raise `ValueError` by default.
- `outcomePrices`: JSON-encoded array `["YES_price", "NO_price"]`. `"1"` = that outcome
  won. Used to determine `settlement` (1.0 or 0.0).
- `endDate`: ISO-8601 UTC, determines the settlement timestamp `T`.

**Alternative: auto-generated JSONL from real data**

When using `--use-real-data` in `prepare_event_model.py`, the JSONL file is generated
automatically by `real_data_builder.write_contracts_jsonl()` from the parquet. The
output `real_contracts.jsonl` uses the same schema and can be loaded by the strategy
via the `contracts_jsonl` config key.

---

## 3. Feature Engineering

Features are computed by the functions in `polymarket/event_features.py`. All 15
features are strictly causal — they use only data available at bar timestamp `t`.

### Contract features (4)

These features describe the contract's current state relative to BTC price.

| Feature | Formula | Intuition |
|---|---|---|
| `log_moneyness` | `log(BTC_close_t / K)` | Positive = above strike, negative = below |
| `log_h_remaining` | `log(max(hours_to_T, 0.5))` | Time-to-expiry on log scale; clipped at 0.5h |
| `sigma_h` | `σ_1h × √(h_remaining)` | Expected BTC move to expiry (lognormal σ) |
| `bs_prob` | `Φ(log_moneyness / sigma_h)` | Black-Scholes probability (driftless) — used as a feature, not the prediction |

The Black-Scholes probability is included as a feature (not the output) because it is
a natural baseline. The model learns to correct this prior based on momentum and
volatility regime signals.

### BTC price features (8)

| Feature | Formula | Lookback |
|---|---|---|
| `log_ret_1h` | `log(Close_t / Close_{t-1})` | 1 bar |
| `log_ret_24h` | `log(Close_t / Close_{t-24})` | 24 bars |
| `log_ret_7d` | `log(Close_t / Close_{t-168})` | 168 bars |
| `realized_vol_24h` | `std(log_ret_1h, window=24, min=12)` | 24 bars |
| `realized_vol_7d` | `std(log_ret_1h, window=168, min=48)` | 168 bars |
| `vol_ratio` | `realized_vol_24h / realized_vol_7d` | — |
| `rsi_14` | RSI with 14-period Wilder smoothing | 15 bars |
| `momentum_7d` | `Close_t / Close_{t-168} − 1` | 168 bars |

Rolling windows always respect time ordering. No shuffling is ever applied to the
data.

### On-chain features (3)

| Feature | Source column | Lag |
|---|---|---|
| `mvrv_lag24` | `mvrv` | 24 hours |
| `hash-rate_lag24` | `hash-rate` | 24 hours |
| `difficulty_lag24` | `difficulty` | 24 hours |

On-chain data is intentionally lagged 24 hours. This models the realistic scenario
where on-chain metrics are published with a reporting delay. If the source columns are
absent from the CSV, the corresponding feature columns will be NaN and are effectively
masked out by the model (treated as zero-weight features after the logistic regression
is fit on non-NaN training data).

### Excluded features

**Market capitalisation** is excluded because it equals `BTC_price × supply` — it is
a deterministic function of the Close price and adds no independent information.

---

## 4. Label Construction

The label is:

```python
label = int(BTC_Open_at_T > K)
```

where:
- `BTC_Open_at_T` is the **open price** of the hourly bar at settlement time `T`.
- `K` is the contract strike.

### Why Open at T

Polymarket resolves BTC contracts using the Binance 1-minute candle close at noon ET
(12:00 US/Eastern). In January, noon ET = 17:00 UTC. The hourly bar that opens at
17:00 UTC has its open price equal to the BTC price at that exact moment, which closely
approximates the 1-minute candle close.

### Label is constant within an event

Every hourly observation for the same (K, T) pair receives the same label. The label
is determined at training time from historical data. For training purposes, it is the
known historical outcome.

### What one "event" means

One event is one (strike K, settlement time T) pair. A single event contributes
`window_days × 24` rows to the training data (one per hourly bar in the 7-day window
before settlement). Features vary across rows; the label is constant.

### Training data composition

With default settings:

- Weekly settlements: Mondays at 17:00 UTC
- Date range: 2018-01-01 to 2025-06-01 ≈ 387 weekly settlement times
- Strikes per event: 7 multipliers × reference price, rounded to nearest $1,000
- Rows per event-strike: 7 days × 24 hours = 168
- Total rows: 387 events × 7 strikes × ~168 bars = ~454,000 (minus feature warmup
  gaps early in the dataset) ≈ 363,000+ rows in practice

---

## 5. Temporal Split and Leakage Prevention

### The core rule

**All splits are by event settlement time T, not by observation timestamp `t`.**

This is critical. A single event generates 168 hourly rows. If you split by row index
or row timestamp, rows from the same event could appear in both train and test,
guaranteeing leakage (the test rows would share the same label as the training rows for
the same event).

### Split structure

```
2018-01-01 ──────────────── 2024-01-01 ──────── midpoint ──────── 2025-06-01
│                            │                    │                 │
│◄────── Training ──────────►│◄── Calibration ───►│◄─── Held-out ──►│
│    T < val_cutoff           │  first 50% of val  │ second 50% of val│
│    n ≈ 277,200              │  n ≈ 43,298        │ n ≈ 43,299      │
```

- **Training set** (T < `val_cutoff`): used to fit model weights.
- **Calibration fold** (first half of T ≥ `val_cutoff`): used to fit the isotonic
  calibrator on top of the trained model's raw scores. **The base model is not retrained
  on this data.**
- **Held-out fold** (second half of T ≥ `val_cutoff`): never seen during fitting or
  calibration; used only for final evaluation metrics.

### How to avoid leakage

✅ **Do:** Use `val_cutoff` as the split boundary and set it to a date well before any
contracts you plan to trade.

✅ **Do:** Ensure `end_date` (the last settlement date for training data) is strictly
before the live deployment date.

✅ **Do:** Run inference on BTC data in temporal order; never batch-shuffle before
feature computation.

❌ **Don't:** Split by row index or `dt` (observation time). An event with T=2024-03-04
has observation rows from 2024-02-26 to 2024-03-04. Splitting at 2024-01-01 by `dt`
would still include those rows in training since `dt` < `T`.

❌ **Don't:** Use any future price (relative to `dt`) in feature computation. The
`add_btc_features` and `add_contract_features` functions enforce this through causal
rolling windows.

❌ **Don't:** Use settlement outcome as a feature. The `label` column is excluded from
`ALL_FEATURE_COLS`.

---

## 6. Model Training and Calibration

### Supported model types

#### `logistic` (default, recommended)

- Pipeline: `StandardScaler → LogisticRegression`
- Parameters: `max_iter=2000`, `C=1.0`, `solver="lbfgs"`, `random_state=42`
- Appropriate for datasets of any size; well-calibrated by construction.
- Interpretable: log-odds weights can be inspected to understand feature importance.
- Recommended for production use and first-pass validation.

#### `xgboost`

- Pipeline: `StandardScaler → XGBClassifier`
- Parameters: `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`,
  `subsample=0.8`, `colsample_bytree=0.8`
- May capture non-linear interactions but requires larger calibration sets.
- Recommended when you have 50+ distinct events in the calibration fold.
- Slower to train; may overfit on small datasets.

### Calibration

Raw classifier scores (log-odds or probability estimates) are mapped to empirical
probabilities using **isotonic regression** on the calibration fold:

```python
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_scores_on_calib, y_calib)
calibrated_prob = iso.transform(raw_score_on_new_data)
```

Isotonic regression fits a non-decreasing step function from raw scores to true outcome
rates. It is monotone by construction, which ensures that the calibrated probabilities
can be used directly for Kelly bet sizing without further adjustment.

The `_CalibratedModel` wrapper stores the base pipeline and isotonic calibrator together
in the saved pickle.

### Reported metrics

After training, metrics are printed and stored in `model_package["metrics"]`:

```python
{
  "train": {"auc": float, "brier": float, "accuracy": float, "n": int},
  "calib": {"auc": float, "brier": float, "accuracy": float, "n": int},
  "held":  {"auc": float, "brier": float, "accuracy": float, "n": int},
}
```

All metrics are computed using the **calibrated** model's output probabilities.

### Current model metrics

Trained on `data_1h.csv` from 2018-01-01 to 2025-06-01 with `val_cutoff=2024-01-01`:

| Split | AUC | Brier | Accuracy | Samples |
|---|---|---|---|---|
| Training | 0.9588 | 0.0779 | 89.66% | 277,200 |
| Calibration | 0.9723 | 0.0661 | 90.46% | 43,298 |
| **Held-out** | **0.9678** | **0.0722** | **90.11%** | 43,299 |

---

## 7. Inference on New Contracts

To generate predictions for a new set of Polymarket contracts using the existing trained
model:

### Step 1 — Prepare contract metadata

Create a JSONL file (see [Section 2](#2-input-data-requirements)) for your contracts.

### Step 2 — Extend BTC data

Ensure `data_1h.csv` contains all hourly bars from at least 7 days before the first
contract's start date through the contract's settlement time.

### Step 3 — Call `build_event_predictions`

```python
from polymarket.contracts import load_contracts
from polymarket.data_builder import build_event_predictions

contracts = load_contracts("user_data/data/polymarket_contracts/feb17.jsonl")

build_event_predictions(
    btc_csv_path="mycode/data/data_1h.csv",
    model_path="user_data/data/polymarket_ml/event_model.pkl",
    contracts=contracts,
    output_dir="user_data/data/polymarket_ml",
)
```

Or equivalently, run the prepare script with `--skip-training-data` to reuse the
existing training data:

```bash
python scripts/prepare_event_model.py \
    --contracts user_data/data/polymarket_contracts/feb17.jsonl \
    --skip-training-data
```

### Step 4 — Update the config

Add the new pairs to `pair_whitelist` in `config_polymarket_ml.json`.

### What the model needs at inference time

The model requires the same 15 features that were used during training, computed at
each hourly bar before the contract's settlement time. Specifically, the minimum
BTC history needed is:

```
Rolling lookback: 168 bars (7 days) for 7-day features
Warmup: ~12 bars minimum for 24h realised vol

Total minimum history before first prediction: ~180 hours (~7.5 days)
```

The `predict_contract_probs` function handles this automatically: it drops NaN rows
from the feature matrix before inference.

### Assumption: training distribution

The model was trained on weekly contracts with strikes spanning ±15% of the prevailing
BTC price. It should generalise to contracts with similar characteristics. Extrapolation
to:
- Contracts more than ~20% out-of-the-money (moneyness_std > 2.0)
- Settlement horizons shorter than 12 hours or longer than 14 days
- Non-weekly settlement schedules

…may produce unreliable probabilities. Check the model's `bs_prob` feature against
the output probability as a sanity check: for deep OTM contracts, the two should agree
roughly.

---

## 8. Temporal Usage Rules

This section summarises the time-ordering constraints that must be respected throughout
the pipeline.

### What "current features at time t" means

At any bar with timestamp `t`, the feature vector contains **only information that was
observable at or before `t`**:

- `Close` prices at `t`, `t−1h`, `t−24h`, `t−168h` ✅
- Rolling volatility computed over past bars ✅
- On-chain data shifted 24h backward ✅
- BTC price at settlement time `T` (future) ❌
- Label (outcome) of the contract (future) ❌

### What settlement time T means

`T` is the fixed timestamp at which the binary contract resolves. For the Jan-20 2026
contracts, `T = 2026-01-20T17:00:00Z`.

The feature vector at time `t` (where `t < T`) includes:
- `log_h_remaining = log(hours between t and T)` — decreases as t approaches T.
- `sigma_h = σ_1h × √(hours_between_t_and_T)` — shrinks as expiry approaches.
- `bs_prob` — updates every hour as moneyness and time-to-expiry change.

### How the model is used on a live date

On a live date, you call `predict_contract_probs(btc_df, K, T, model)` with:
- `btc_df`: all available BTC hourly bars up to the current moment
- `K`, `T`: the contract's strike and settlement time
- `model`: the saved model package

The function returns a probability for every past bar. The **most recent** row's
`fair_value` is the current model estimate of P(BTC_T > K).

### What not to do

| Action | Why it is wrong |
|---|---|
| Shuffle rows before feature computation | Destroys temporal order; creates lookahead in rolling features |
| Use `T < val_cutoff` events in the calibration fold | Calibration data must be after training data by event time |
| Include the settlement candle itself as a training feature | The settlement price is the label — using it as a feature is target leakage |
| Split training data by observation time `t` instead of event time `T` | Same event spans 168 rows; splitting by `t` leaks labels |
| Compute volatility including bar `t+1` | Lookahead in rolling windows |

### Using the model on future contracts

The trained model does not expire. As long as:
1. The BTC price series covers the required lookback window, and
2. The new contracts have similar characteristics to the training distribution,

the model can be applied to contracts from any future date without retraining.

The model's `val_cutoff` (2024-01-01) is relevant only to the training/validation split
used during fitting. It does not limit when the model can be deployed.

---

## 9. Training Artefacts

After running `prepare_event_model.py`, the following files are written to
`user_data/data/polymarket_ml/`:

**Synthetic mode (default):**

| File | Format | Description |
|---|---|---|
| `event_model_training.parquet` | Parquet | 363K+ training samples; all 15 features + label, K, T |
| `event_model.pkl` | joblib pickle | Model package dict (see below) |
| `{pair}-event_probs.csv` | CSV | Per-contract fair-value predictions; columns: `dt_utc`, `fair_value` |
| `{pair}-1h.feather` | Feather | Synthetic OHLCV for backtester (generated by `build_all_feathers`) |

**Real-data mode (`--use-real-data`):** Same artefacts, plus:

| File | Format | Description |
|---|---|---|
| `{pair}-1h.feather` | Feather | Real Polymarket YES-side OHLCV, 7-day window before expiry |
| `real_contracts.jsonl` | JSONL | Auto-generated contract metadata; compatible with `load_contracts()` |

### Model package structure

The joblib pickle contains a Python dict:

```python
{
    "model":        _CalibratedModel,    # base Pipeline + IsotonicRegression wrapper
    "feature_cols": list[str],           # canonical list of 15 feature names
    "val_cutoff":   "2024-01-01",        # training cutoff date (informational)
    "model_type":   "logistic",          # "logistic" or "xgboost"
    "metrics": {
        "train": {"auc": 0.9588, "brier": 0.0779, "accuracy": 0.8966, "n": 277200},
        "calib": {"auc": 0.9723, "brier": 0.0661, "accuracy": 0.9046, "n": 43298},
        "held":  {"auc": 0.9678, "brier": 0.0722, "accuracy": 0.9011, "n": 43299},
    },
}
```

### Loading the model

```python
from polymarket.event_model import load_model, predict_contract_probs
import pandas as pd

model_package = load_model("user_data/data/polymarket_ml/event_model.pkl")

# Generate predictions for a contract
btc_df = ...  # DataFrame from load_btc_hourly()
K = 90_000.0
T = pd.Timestamp("2026-01-20T17:00:00Z", tz="UTC")

probs_df = predict_contract_probs(btc_df, K, T, model_package)
# probs_df columns: dt_utc, fair_value
```

---

## 10. Retraining Guidance

### When to retrain

| Trigger | Recommended action |
|---|---|
| New BTC data available (quarterly) | Run `prepare_event_model.py` with extended `--end-date` |
| Held-out AUC drops below 0.90 | Retrain immediately; investigate features |
| Market regime shift (e.g. post-halving, macro shock) | Retrain with updated `--start-date` or narrower window |
| More than 12 months since last training | Retrain as a routine maintenance task |
| Deploying on contracts with substantially different strike ranges | Retrain with matching synthetic strike distribution |

### Recommended retraining cadence

**Minimum:** Annually, using all available history (extend `--end-date`).

**Preferred:** Quarterly, rolling `--end-date` forward to include the most recent 3
months of data.

Example for quarterly refresh in Q2 2025:

```bash
python scripts/prepare_event_model.py \
    --start-date 2018-01-01 \
    --end-date   2025-06-01 \
    --val-cutoff 2024-06-01
```

### Trade-offs

| Approach | Benefit | Risk |
|---|---|---|
| Retrain on longer history | More events, higher AUC, stable calibration | May include stale market regime |
| Retrain on recent data only (rolling window) | Better regime alignment | Fewer events; AUC may degrade |
| Fixed model, no retraining | Stable; no refit risk | Drift; degrading calibration |

For the logistic regression model, retraining is fast (< 1 minute on a laptop for 363K
samples). There is little cost to retraining frequently.

### Signs that retraining is needed

- Consecutive contracts where model edge > 0.04 but outcomes are systematically wrong
  in one direction (e.g. consistently predicting YES but NO resolves).
- Held-out calibration plot shows model probabilities systematically higher or lower
  than realised rates.
- BTC market enters a regime not represented in the training data (e.g. post-ETF
  approval dynamics, new regulatory environment).

### Avoiding overfitting on small validation sets

When training on short date ranges (< 2 years), the calibration and held-out folds may
contain fewer than 20 distinct events. In this case:
- Use `--model-type logistic` (not xgboost).
- Inspect held-out metrics carefully; AUC from very small sets is noisy.
- Consider combining multiple contract batches (different settlement dates) into the
  validation window to increase held-out event count.

---

## 11. Extending Predictions

### Adding more feature columns to event_probs.csv

The current `build_event_predictions` writes only `dt_utc` and `fair_value` to the
per-contract CSV. To enable the OTM penalty in `EventProbAlpha`, also write
`log_moneyness` and `sigma_h`.

Modify `polymarket/data_builder.py` in `build_event_predictions`:

```python
# After computing probs_df from predict_contract_probs():
# Re-compute features to get log_moneyness and sigma_h for the same rows
from polymarket.event_features import add_btc_features, add_contract_features

df_feat = add_btc_features(btc_df)
df_feat = add_contract_features(df_feat, K, T)
df_feat = df_feat[df_feat["dt"] < T].dropna(subset=["log_moneyness", "sigma_h"])

probs_df = probs_df.merge(
    df_feat[["dt", "log_moneyness", "sigma_h"]].rename(columns={"dt": "dt_utc"}),
    on="dt_utc",
    how="left",
)
```

With these columns present, `EventProbAlpha` will automatically apply the OTM penalty
without any other code changes.

### Adding a new feature

1. Add the feature computation to `add_btc_features` or `add_contract_features` in
   `polymarket/event_features.py`.
2. Add the column name to `BTC_FEATURE_COLS` or `CONTRACT_FEATURE_COLS`.
3. Retrain the model — `ALL_FEATURE_COLS` is automatically updated.
4. Regenerate predictions with `prepare_event_model.py`.

Feature additions do not require any changes to `EventProbAlpha` or the strategy.

### Using XGBoost instead of logistic regression

```bash
python scripts/prepare_event_model.py \
    --model-type xgboost \
    --skip-training-data
```

This reuses the existing training Parquet and retrains only the model. Compare the
held-out AUC and Brier score against the logistic baseline before switching.

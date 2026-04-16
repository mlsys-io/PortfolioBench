# Polymarket BTC Event-Probability Strategy

This document describes the Polymarket BTC binary-contract trading pipeline built into
PortfolioBench. It covers the full workflow: contract data → model → predictions →
backtest, as well as practical guidance on running and extending the pipeline.

For the technical training and inference reference, see
[training-guide.md](training-guide.md).

---

## Table of Contents

1. [What This Pipeline Does](#1-what-this-pipeline-does)
2. [Quick Start — Using the Pre-Trained Model](#2-quick-start--using-the-pre-trained-model)
3. [Complete Pipeline — Training from Scratch](#3-complete-pipeline--training-from-scratch)
4. [Strategy Reference](#4-strategy-reference)
5. [Backtesting](#5-backtesting)
6. [Applying to Other Contract Dates](#6-applying-to-other-contract-dates)
7. [Model Quality Metrics](#7-model-quality-metrics)
8. [Practical Limitations and Caveats](#8-practical-limitations-and-caveats)

---

## 1. What This Pipeline Does

### The problem

Polymarket runs weekly BTC binary contracts of the form:

> "Will the price of Bitcoin be above $90,000 on January 20?"

Each contract settles to $1 (YES wins) or $0 (NO wins) based on whether BTC's price at
the specified settlement time crosses the strike.

A naïve trading model faces a structural misalignment: general-purpose BTC direction
classifiers predict whether price goes *up or down*, but the contract pays off based on
whether price is *above or below a specific level at a specific time*. These are
different questions.

### The solution

The pipeline trains a **direct event-probability model** that answers:

> P(BTC_T > K | BTC price series and market context at time t)

where:
- `K` is the contract strike in USD
- `T` is the settlement timestamp
- `t` is the current time (strictly before `T`)

This probability is the contract's fair value. When the fair value materially exceeds
the market price (the contract's current quote), there is positive edge. The strategy
trades only when that edge is large enough to justify the risk.

### Why this framing is better aligned with the contract structure

| Aspect | Indirect approach | Direct approach |
|---|---|---|
| What is predicted | Up/down direction | P(BTC > K at T) |
| Contract payoff alignment | Indirect (requires BS bridge) | Direct |
| Strike specificity | Ignores strike | Strike-aware (log-moneyness feature) |
| Horizon specificity | Fixed horizon | Hours-remaining feature |
| Calibration | Requires two-stage conversion | Single-stage isotonic calibration |

The direct model learns from 363,000+ historical BTC binary event samples spanning
2018–2024, covering diverse market regimes (bull runs, crashes, sideways periods).

---

## 2. Quick Start — Using the Pre-Trained Model

A trained model and pre-computed predictions for the Jan-20 2026 contracts are already
included in the repository.

### Prerequisites

```bash
git clone --recurse-submodules https://github.com/mlsys-io/PortfolioBench.git
cd PortfolioBench
pip install -e .
```

### Run the backtest immediately

```bash
portbench backtesting \
    --strategy DualModelPolymarketPortfolio \
    --strategy-path ./user_data/strategies \
    --config user_data/config_polymarket_ml.json \
    --datadir user_data/data/polymarket_ml \
    --timerange 20260113-20260121
```

### What you should see

```
Total profit %    :  +1.11%
Final balance     :  10,110.69 USDT
Trades            :  1  (1 win / 0 loss)
Drawdown          :  0.00%
Market change     : -51.12%
```

The strategy opened one position (BTCABOVE88K-JAN20-YES) and held it to settlement.
The contract resolved YES (BTC was above $88K at expiry), returning +13.85% on the
position. The portfolio gained +$110 on a $10,000 starting balance despite the
underlying market dropping 51%.

### Pre-built artefacts

| File | Description |
|---|---|
| `user_data/data/polymarket_ml/event_model.pkl` | Trained + calibrated logistic regression model |
| `user_data/data/polymarket_ml/event_model_training.parquet` | 363K training samples (2018–2025) |
| `user_data/data/polymarket_ml/BTCABOVE*-event_probs.csv` | Hourly fair-value predictions per contract |
| `user_data/data/polymarket_ml/BTCABOVE*.feather` | Synthetic OHLCV price series per contract |
| `user_data/data/polymarket_contracts/jan20.jsonl` | Contract metadata (strike, expiry, settlement) |

---

## 3. Complete Pipeline — Training from Scratch

Run the preparation script to rebuild everything from the raw BTC price series:

```bash
python scripts/prepare_event_model.py
```

This executes four steps automatically:

| Step | What it does | Output |
|---|---|---|
| 0 — Build feather files | Generates OHLCV feather files per contract (synthetic by default; see [Using Real Data](#using-real-polymarket-data)) | `*.feather` files in `--output-dir` |
| 1 — Build training data | Constructs synthetic weekly BTC binary events from `data_1h.csv` | `event_model_training.parquet` |
| 2 — Train model | Fits and calibrates a logistic regression on the training data | `event_model.pkl` |
| 3 — Generate predictions | Runs the model at every hourly bar before each contract's expiry | Per-contract `*-event_probs.csv` files |

### Full CLI flags

```bash
python scripts/prepare_event_model.py \
    --btc-csv      mycode/data/data_1h.csv \
    --contracts    user_data/data/polymarket_contracts/jan20.jsonl \
    --output-dir   user_data/data/polymarket_ml \
    --start-date   2018-01-01 \
    --end-date     2025-06-01 \
    --val-cutoff   2024-01-01 \
    --model-type   logistic
```

| Flag | Default | Description |
|---|---|---|
| `--btc-csv` | `mycode/data/data_1h.csv` | Path to hourly BTC OHLCV CSV |
| `--contracts` | `user_data/data/polymarket_contracts/jan20.jsonl` | Contract metadata JSONL (ignored when `--use-real-data` is set) |
| `--output-dir` | `user_data/data/polymarket_ml` | Where all artefacts are written |
| `--start-date` | `2018-01-01` | Earliest settlement date for training samples |
| `--end-date` | `2025-06-01` | Latest settlement date (exclusive) |
| `--val-cutoff` | `2024-01-01` | Events before this date → training; after → validation |
| `--model-type` | `logistic` | `logistic` (default) or `xgboost` |
| `--skip-feathers` | off | Skip step 0 if feather files already exist in `--output-dir` |
| `--skip-training-data` | off | Skip step 1 if the Parquet already exists |
| `--use-real-data` | off | Step 0: build feathers from real Polymarket trade data instead of synthetic prices |
| `--parquet-path` | `mycode/data/combined_filtered_data.paquet` | Path to the Polymarket trade-history parquet (only used with `--use-real-data`) |

### Using Real Polymarket Data

By default, step 0 generates **synthetic** OHLCV price series for each contract. If you
have a Polymarket trade-history parquet file, you can use real market prices instead:

```bash
python scripts/prepare_event_model.py \
    --use-real-data \
    --parquet-path mycode/data/combined_filtered_data.paquet \
    --output-dir   user_data/data/polymarket_ml_real
```

This runs `polymarket/real_data_builder.py` which:

1. Loads the parquet and filters to BTC/Bitcoin YES-side rows.
2. Parses each market's strike (`$88K`, `$90,000`, etc.) and expiry date from the
   `question` text.
3. Checks that the last 7-day window has at least 60% hourly coverage (≥ 101 of 168
   candles).
4. Forward-fills gaps of up to 6 consecutive hours; excludes contracts with longer gaps.
5. Writes a freqtrade-compatible `*.feather` file for each qualifying contract.
6. Serialises all parsed contracts to `real_contracts.jsonl` in the output directory.

The event model and downstream prediction steps (steps 1–3) are unchanged — they work
with the same feather files regardless of whether they were generated synthetically or
from real data.

**Parquet format requirements:**

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime (UTC) | Candle open time |
| `condition_id` | string | Market identifier |
| `side` | string | `"Yes"` or `"No"` — only YES-side rows are used |
| `open`, `high`, `low`, `close` | string-encoded float | OHLC prices (0–1 range) |
| `volume` | int64 | Traded volume |
| `question` | string | Human-readable market question |

**Coverage thresholds:**

| Gap length | Treatment |
|---|---|
| 1–6 hours | Forward-filled (price unchanged) |
| 7–23 hours | Forward-filled, but a `WARNING` is logged |
| ≥ 24 hours consecutively | Contract excluded (price too stale) |
| < 60% of window rows present | Contract excluded |

For the full training reference — feature engineering, label construction, temporal
splits, calibration, and inference — see [training-guide.md](training-guide.md).

---

## 4. Strategy Reference

The strategy is implemented in
`user_data/strategies/DualModelPolymarketPortfolio.py`.

### How it works

At each hourly candle, the strategy:

1. **Loads the pre-computed fair value** for the contract from the per-contract CSV.
2. **Computes the edge**: `edge = fair_value − market_price`.
3. **Applies fractional Kelly sizing** to determine how much capital to allocate.
4. **Enters a YES position** when edge exceeds `MIN_EDGE` and Kelly allocation is
   positive.
5. **Holds to expiry** — no stop loss, no ROI exit, no rebalancing.
6. **Settles at expiry** via `custom_exit_price`: returns 0.999 for YES win,
   0.001 for NO win.

### Configurable parameters

| Parameter | Default | Description |
|---|---|---|
| `MIN_EDGE` | `0.04` | Minimum edge (fair\_value − market\_price) required to open a position |
| `KELLY_FRACTION` | `0.15` | Fractional Kelly multiplier — dampens raw Kelly bet size |
| `MAX_ALLOC` | `0.08` | Maximum fraction of capital allocated to any single contract |
| `MAX_EXPIRY_ALLOC` | `0.10` | Maximum fraction of total capital across all contracts with the same settlement date |

To override parameters without modifying the strategy file, subclass
`DualModelPolymarketPortfolio` and set the class attributes.

### Configurable data paths (via config JSON)

Two file paths used by the strategy can be overridden through the freqtrade config JSON
without touching the strategy source:

| Config key | Default | Description |
|---|---|---|
| `contracts_jsonl` | `user_data/data/polymarket_contracts/jan20.jsonl` | Path to the contract metadata JSONL file; relative paths are resolved against `user_data/` |
| `predictions_dir` | `user_data/data/polymarket_ml` | Directory containing per-contract `*-event_probs.csv` files; relative paths are resolved against `user_data/` |

Example config for a real-data backtest:

```json
{
  "contracts_jsonl": "data/polymarket_ml_real/real_contracts.jsonl",
  "predictions_dir": "data/polymarket_ml_real",
  "exchange": {
    "name": "polymarket",
    "pair_whitelist": [
      "BTCABOVE108K-SEP5-YES/USDT",
      "BTCABOVE110K-SEP5-YES/USDT"
    ]
  }
}
```

Contracts loaded from `real_contracts.jsonl` (produced by `--use-real-data`) may
contain question patterns that the minimal regex in the synthetic pipeline did not
support (e.g. "reach $108K"). The loader automatically skips unparseable lines with a
warning, so a mixed JSONL file will not crash the strategy.

### Position sizing (Kelly formula)

For a contract priced at `p_market` and model probability `p_model`:

```
b          = (1 / p_market) − 1      # net odds on a YES win
q          = 1 − p_model             # model probability of NO win
raw_kelly  = (p_model × b − q) / b  # full Kelly fraction
kelly_alloc = min(raw_kelly × KELLY_FRACTION, MAX_ALLOC)
stake       = kelly_alloc × free_capital
```

The stake is then further capped by the per-expiry notional limit.

### OTM distance penalty

Contracts where BTC is below the strike (out-of-the-money for YES contracts) are
penalised to reduce exposure to highly uncertain outcomes. ITM contracts (BTC above
the strike) are not penalised.

The standardised OTM distance is:

```
moneyness_std = max(0, −log(BTC_current / K)) / σ_h
```

where `σ_h = σ_1h × √(hours_remaining)` is the expected BTC move to expiry. Only
negative `log(BTC/K)` (BTC below strike) contributes.

| OTM distance | Effect |
|---|---|
| 0 – 1.5σ | `effective_min_edge = MIN_EDGE × (1 + 2 × moneyness_std)`; `effective_max_alloc = MAX_ALLOC / (1 + moneyness_std)` |
| > 1.5σ | Skipped entirely (zero allocation) |

**Note:** The OTM penalty requires `log_moneyness` and `sigma_h` columns to be present
in the per-contract event probability CSV. The current `build_event_predictions`
pipeline writes only `dt_utc` and `fair_value`, so this penalty is inactive by default.
See [training-guide.md — Extending Predictions](training-guide.md#11-extending-predictions)
for how to enable it.

### Same-expiry correlation guard

Each Polymarket settlement date can have multiple strikes (88K, 90K, 92K…). These are
**not independent bets** — they are all determined by the same underlying BTC terminal
price. Entering all of them simultaneously concentrates risk.

The strategy enforces a **one-position-per-expiry rule**: `confirm_trade_entry` rejects
any contract whose `end_date_utc` matches an already-open position's expiry. Combined
with the `MAX_EXPIRY_ALLOC` cap, total exposure to a single settlement event is bounded
regardless of how many signals fire.

If multiple contracts generate simultaneous signals, the first one processed (lowest
strike in the pair whitelist) wins. To control which contract is preferred, order the
`pair_whitelist` in `config_polymarket_ml.json` with the desired priority at the top.

---

## 5. Backtesting

### Prerequisites

The following files must exist before running a backtest:

```
user_data/data/polymarket_ml/
├── BTCABOVE88K-JAN20-YES_USDT-1h.feather     ← synthetic OHLCV (required by freqtrade)
├── BTCABOVE88K-JAN20-YES_USDT-event_probs.csv ← fair-value predictions (required by strategy)
├── ...                                        (one pair of files per contract)
└── event_model.pkl                            ← trained model (needed only for step 3)

user_data/data/polymarket_contracts/
└── jan20.jsonl                                ← contract metadata
```

Generate all of these with `python scripts/prepare_event_model.py` if they are not
already present.

### Run a backtest

```bash
portbench backtesting \
    --strategy DualModelPolymarketPortfolio \
    --strategy-path ./user_data/strategies \
    --config user_data/config_polymarket_ml.json \
    --datadir user_data/data/polymarket_ml \
    --timerange 20260113-20260121
```

The `--timerange` flag is `YYYYMMDD-YYYYMMDD` and should span the contract window
(typically the week before settlement to the settlement date).

### Config file

The strategy config is at `user_data/config_polymarket_ml.json`. Key settings:

```json
{
  "max_open_trades": 5,
  "stake_amount":    "unlimited",
  "dry_run_wallet":  10000,
  "fee":             0.0,
  "exchange": {
    "name": "polymarket",
    "pair_whitelist": [
      "BTCABOVE88K-JAN20-YES/USDT",
      "BTCABOVE90K-JAN20-YES/USDT",
      "BTCABOVE92K-JAN20-YES/USDT",
      "BTCABOVE94K-JAN20-YES/USDT",
      "BTCABOVE96K-JAN20-YES/USDT"
    ]
  }
}
```

- Add or remove pairs from `pair_whitelist` to control which contracts are traded.
- The order of pairs in `pair_whitelist` determines priority when multiple same-expiry
  signals fire simultaneously.
- `fee: 0.0` models Polymarket's fee structure as zero; adjust if your account incurs
  taker fees.

### Interpreting results

| Metric | What to look for |
|---|---|
| `Total profit %` | Absolute P&L as % of starting balance |
| `Avg profit %` | Per-trade average — a single good trade can dominate |
| `Drawdown` | Should be near zero for a well-sized single binary position |
| `Win %` | Binary contracts settle to either 0 or 1; win% depends heavily on strike selection |
| `Market change` | Context: BTC spot return over the same window |
| `Rejected Entry signals` | Signals rejected by `confirm_trade_entry`; expect 3–4 per settlement date if only one per expiry is allowed |

### What the backtester assumes

- Contracts are entered at the close price of the first candle where `enter_long = 1`.
- Fees are zero (as configured).
- Settlement prices are 0.999 (YES) or 0.001 (NO), not the market price at expiry.
- The feather files contain synthetic OHLCV data (not real Polymarket order book data).
  The open/close prices reflect modelled option-like dynamics and may not match
  historical Polymarket quotes exactly.

---

## 6. Applying to Other Contract Dates

To trade or backtest a different set of Polymarket BTC contracts (e.g., February
expiry, different strikes), you need to supply:

### 1 — Contract metadata JSONL

A JSONL file where each line is a Polymarket market JSON object. The required fields
are:

```json
{
  "question":      "Will the price of Bitcoin be above $95,000 on February 17?",
  "endDate":       "2026-02-17T17:00:00Z",
  "startDate":     "2026-02-10T17:00:00Z",
  "outcomePrices": "[\"1\", \"0\"]",
  "id":            "...",
  "slug":          "..."
}
```

The `load_contracts()` function extracts the strike and direction from the `question`
field. Supported question patterns include:

| Pattern | Direction | Example |
|---|---|---|
| `above $X,XXX` / `above $XXK` | above | "Bitcoin above $90,000 on January 20?" |
| `reach/hit/exceed/surpass $X,XXX` | above | "Will Bitcoin reach $120,000 by December?" |
| `below $X,XXX` / `less than $X,XXX` | below | "Bitcoin below $80,000 on March 1?" |
| `dips to $X,XXX` | below | "Will Bitcoin dip to $70K in October?" |

Strikes can be written with or without a `K`/`k` suffix (e.g. `$88K` = `$88,000`).
Settlement is determined from `outcomePrices`.

Lines whose question cannot be parsed are skipped with a warning (see `skip_unparseable`
in `load_contracts`) rather than raising an error, so a JSONL file with mixed contract
types will not crash the strategy.

Place the file at `user_data/data/polymarket_contracts/<name>.jsonl` and update
`--contracts` in the prepare script.

### 2 — BTC hourly price data

The BTC CSV at `mycode/data/data_1h.csv` must cover:

- At least 7 days **before** the earliest contract start (for rolling feature warmup).
- All hours up to (and including) the contract settlement time.

Expected CSV format:

```
Timestamp,Open,High,Low,Close,Volume
1736780400,91200.00,91500.00,91100.00,91300.00,1234.5
...
```

- `Timestamp`: Unix epoch **seconds** (integer).
- `Close`: BTC/USDT price.
- Optional on-chain columns: `mvrv`, `hash-rate`, `difficulty` (omit if unavailable;
  the corresponding features will be NaN and effectively ignored by the model).

### 3 — Feather files (synthetic or real OHLCV)

The backtester requires an OHLCV feather file for each pair. Two sources are supported:

- **Synthetic (default):** generated by `build_all_feathers()` from `data_builder.py`.
  These are modelled price series used solely to satisfy freqtrade's data loading
  requirements. The strategy ignores OHLCV values other than `close` for market price.
- **Real Polymarket data:** generated by `build_all_feathers_from_parquet()` from
  `real_data_builder.py` (requires a trade-history parquet). These contain actual
  historical Polymarket prices from the YES-side order book. Use `--use-real-data` in
  `prepare_event_model.py` to build these instead of synthetic files.

### 4 — Per-contract predictions

Run step 3 of `prepare_event_model.py` (or call `build_event_predictions()` directly)
with your new contracts JSONL to generate fresh `*-event_probs.csv` files. The trained
model can be reused as long as the BTC price data covers the required window.

### 5 — Config update

Update `user_data/config_polymarket_ml.json` to list the new contract pairs:

```json
"pair_whitelist": [
  "BTCABOVE92K-FEB17-YES/USDT",
  "BTCABOVE95K-FEB17-YES/USDT"
]
```

The pair naming convention is: `BTC{ABOVE|BELOW}{STRIKE_IN_K}-{MON}{DAY}-YES/USDT`
(e.g. `BTCABOVE95K-FEB17-YES/USDT` for BTC above $95K on Feb 17).

---

## 7. Model Quality Metrics

The model is a calibrated logistic regression trained on 363,000+ BTC binary event
samples from 2018-01-01 to 2024-12-31, with temporal hold-out from 2024-01-01 onward.

### Evaluation results

| Split | AUC | Brier | Accuracy | Samples |
|---|---|---|---|---|
| **Training** | 0.9588 | 0.0779 | 89.66% | 277,200 |
| **Calibration** | 0.9723 | 0.0661 | 90.46% | 43,298 |
| **Held-out** | **0.9678** | **0.0722** | **90.11%** | 43,299 |

The held-out split covers events from 2024-07-01 to 2025-06-01 (post-training, post-
calibration). These samples were never seen during model fitting.

### What these metrics mean

**AUC (Area Under the ROC Curve):**
An AUC of 0.97 means the model correctly ranks a random YES outcome above a random NO
outcome 97% of the time. An uninformative model scores 0.50. Values above 0.90 are
considered very strong for a financial prediction task.

**Brier score** (lower is better, perfect = 0.0):
The Brier score measures mean squared error between predicted probability and actual
outcome. A score of 0.072 is well below the baseline of 0.25 (a model that always
predicts 0.5). This indicates the predicted probabilities are close to the true
empirical rates.

**Accuracy at threshold 0.5:**
90.1% correct binary classification on held-out data. Note: the model is designed to
produce calibrated probabilities, not binary predictions — accuracy at 0.5 is a
sanity-check metric rather than the primary optimisation target.

### Why AUC is high

The model benefits from strong structure in the data: contracts far out-of-the-money
(e.g. BTC above $96K when BTC is trading at $78K) are easy to classify NO, and vice
versa. The meaningful difficulty lies in near-the-money contracts in the last 24–48
hours before expiry. Users should interpret the AUC in light of this distribution.

---

## 8. Practical Limitations and Caveats

### What the framework guarantees

- Causal feature engineering: no future price data is used for any historical sample.
- Temporal train/val splits: no future events leak into the training set.
- Isotonic calibration: model probabilities are monotonically corrected to match
  empirical outcome rates on held-out calibration data.

### Assumptions

- **BTC settlement price**: The model uses the open price of the hourly BTC candle at
  the settlement timestamp (17:00 UTC in January) as the resolution price. This
  approximates the Polymarket resolution rule (Binance 1-minute close at noon ET).
  Small discrepancies are possible if the 1-minute candle deviates from the hourly open.
- **Synthetic OHLCV**: Feather files used for backtesting contain modelled, not
  historical, Polymarket prices. Backtest P&L reflects settlement outcomes and model
  probabilities, not actual historical bid-ask dynamics.
- **No slippage or liquidity modelling**: Entry at close price; zero fees. Real
  Polymarket trades may face wider spreads, especially on less-liquid contracts.
- **Single settlement time**: The pipeline is calibrated for weekly contracts settling
  at 17:00 UTC on Mondays (January timezone offset). Contracts with different settlement
  times require re-verification of the resolution price logic.

### Data requirements for live use

Before deploying on a live or forward-tested date:

1. `data_1h.csv` must include all hourly bars up to the present.
2. At minimum, 7 days of BTC price history before the contract start date must be
   available for rolling feature computation.
3. On-chain features (`mvrv`, `hash-rate`, `difficulty`) are optional but improve
   accuracy. If omitted, they default to NaN and contribute zero signal.

### Likely failure modes

| Failure mode | Cause | Mitigation |
|---|---|---|
| No entry signals fired | Edge too low (all contracts priced near fair value) | Check `ml_edge` column; consider lowering `MIN_EDGE` cautiously |
| All contracts blocked by expiry gate | First contract opened; others rejected | Expected behaviour; use `pair_whitelist` ordering to control which contract is preferred |
| Empty prediction CSV | BTC data does not cover the contract window | Extend `data_1h.csv` to include the contract period |
| Model underperforms on new strikes/horizons | Distribution shift from training | Retrain with more recent data; see [training-guide.md](training-guide.md#10-retraining-guidance) |
| Overfit to training period | Small held-out sample relative to training | Monitor held-out AUC; retrain annually at minimum |

### What this framework does not guarantee

- The strategy is profitable on all contract dates. The Jan-20 2026 backtest shows
  +1.11% on one contract. A single-week window is insufficient for statistical
  significance.
- Predictions for contracts with strikes outside the training distribution (e.g.
  far-from-money contracts not represented in the 2018–2025 synthetic data) may be
  poorly calibrated.
- Forward performance will depend on BTC market regime. The model was trained primarily
  on a bull-market period (2020–2024) with some bear-market coverage (2018–2019,
  2022). Performance in regimes not well-represented in the training data is unknown.

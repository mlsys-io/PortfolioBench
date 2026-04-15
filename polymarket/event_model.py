"""Train, save, load, and predict with the direct event-probability model.

The model predicts P(BTC_T > K | features at time t) directly, replacing the
indirect chain: direction_classifier → Black-Scholes → contract probability.

Two model types are supported:

* ``'logistic'`` — LogisticRegression with StandardScaler (MVP; well-calibrated
  by construction; interpretable; safe for small datasets).
* ``'xgboost'``  — XGBClassifier wrapped with isotonic CalibratedClassifierCV
  (stronger but needs larger calibration sets; ~50+ distinct events recommended).

Calibration
-----------
Both model types are calibrated using ``CalibratedClassifierCV(cv='prefit')``
on a held-out calibration fold (the first half of the validation period).
This converts raw scores to empirical probabilities, which are required for
honest Kelly sizing.

Temporal split
--------------
All splits respect event time ``T`` to prevent future events leaking into
the training set.  Row-level shuffling is never used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polymarket.event_features import (
    ALL_FEATURE_COLS,
    add_btc_features,
    add_contract_features,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration wrapper
# ---------------------------------------------------------------------------

class _CalibratedModel:
    """Thin wrapper: base classifier → isotonic calibrator → calibrated proba.

    Decoupled from sklearn's ``CalibratedClassifierCV`` to avoid version
    compatibility issues with the ``cv='prefit'`` parameter removed in
    sklearn 1.8.
    """

    def __init__(self, base: Pipeline, calibrator: IsotonicRegression) -> None:
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        calibrated = self.calibrator.transform(raw)
        return np.column_stack([1.0 - calibrated, calibrated])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    samples: pd.DataFrame,
    val_cutoff: str,
    model_type: str = "logistic",
) -> dict[str, Any]:
    """Train and calibrate the event probability model.

    Temporal split by event time ``T``:

    * ``T < val_cutoff``  → training set (model weights).
    * ``T >= val_cutoff`` → split 50/50 into calibration fold and held-out fold.

    Args:
        samples:    DataFrame from
                    :func:`~polymarket.event_dataset.build_training_samples`.
                    Must have columns ``[*ALL_FEATURE_COLS, 'label', 'T']``.
        val_cutoff: ISO date string.  Events before this date go to training.
        model_type: ``'logistic'`` or ``'xgboost'``.

    Returns:
        Dict with keys:

        * ``model``        — fitted :class:`~sklearn.calibration.CalibratedClassifierCV`.
        * ``feature_cols`` — ``list[str]`` matching :data:`~polymarket.event_features.ALL_FEATURE_COLS`.
        * ``val_cutoff``   — str, as supplied.
        * ``metrics``      — nested dict with train / calib / held-out AUC,
                             Brier score, and accuracy.
        * ``model_type``   — str.

    Raises:
        ValueError: If either the training or validation split is empty.
    """
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    cutoff_ts = pd.Timestamp(val_cutoff, tz="UTC")
    train_mask = samples["T"] < cutoff_ts
    val_mask = samples["T"] >= cutoff_ts

    X_train = samples.loc[train_mask, ALL_FEATURE_COLS].values.astype(float)
    y_train = samples.loc[train_mask, "label"].values.astype(int)
    X_val = samples.loc[val_mask, ALL_FEATURE_COLS].values.astype(float)
    y_val = samples.loc[val_mask, "label"].values.astype(int)

    if len(X_train) == 0:
        raise ValueError(
            f"Training split is empty (val_cutoff={val_cutoff}).  "
            f"Earliest T in samples: {samples['T'].min()}"
        )
    if len(X_val) == 0:
        raise ValueError(
            f"Validation split is empty (val_cutoff={val_cutoff}).  "
            f"Latest T in samples: {samples['T'].max()}"
        )

    logger.info(
        "Training event model [%s]: %d train / %d val samples",
        model_type,
        len(X_train),
        len(X_val),
    )

    # ---- Build base estimator ----
    if model_type == "logistic":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                C=1.0,
                solver="lbfgs",
                random_state=42,
            )),
        ])
    elif model_type == "xgboost":
        import xgboost as xgb
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="auc",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )),
        ])
    else:
        raise ValueError(
            f"Unknown model_type={model_type!r}.  Use 'logistic' or 'xgboost'."
        )

    # ---- Fit base model on training set ----
    base.fit(X_train, y_train)

    # ---- Calibration: split val 50/50 into calib + held-out ----
    calib_n = len(X_val) // 2
    X_calib, X_held = X_val[:calib_n], X_val[calib_n:]
    y_calib, y_held = y_val[:calib_n], y_val[calib_n:]

    # Isotonic regression calibration: fits a monotone mapping from raw
    # classifier scores to empirical outcome rates on the calibration fold.
    raw_calib_scores = base.predict_proba(X_calib)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_calib_scores, y_calib)
    cal_model = _CalibratedModel(base, iso)

    # ---- Metrics ----
    def _compute_metrics(tag: str, X: np.ndarray, y: np.ndarray) -> dict:
        if len(y) == 0:
            return {}
        p = cal_model.predict_proba(X)[:, 1]
        has_both_classes = len(np.unique(y)) > 1
        auc = float(roc_auc_score(y, p)) if has_both_classes else float("nan")
        brier = float(brier_score_loss(y, p))
        acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
        logger.info(
            "  %-6s  AUC=%.4f  Brier=%.4f  Acc=%.4f  n=%d",
            tag, auc, brier, acc, len(y),
        )
        return {"auc": auc, "brier": brier, "accuracy": acc, "n": len(y)}

    logger.info("Evaluation metrics:")
    metrics = {
        "train": _compute_metrics("TRAIN",  X_train, y_train),
        "calib": _compute_metrics("CALIB",  X_calib, y_calib),
        "held":  _compute_metrics("HELD",   X_held,  y_held),
    }

    return {
        "model": cal_model,
        "feature_cols": list(ALL_FEATURE_COLS),
        "val_cutoff": val_cutoff,
        "metrics": metrics,
        "model_type": model_type,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model_package: dict, path: str | Path) -> None:
    """Save the model package to a joblib pickle."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, str(out))
    logger.info("Event model saved → %s", out)


def load_model(path: str | Path) -> dict:
    """Load an event model package from a joblib pickle.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Event model not found: {p}\n"
            "Run scripts/prepare_event_model.py to build it."
        )
    return joblib.load(str(p))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_contract_probs(
    btc_df: pd.DataFrame,
    K: float,
    T: pd.Timestamp,
    model_package: dict,
) -> pd.DataFrame:
    """Predict P(BTC_T > K) for every hourly bar in ``btc_df`` before ``T``.

    Features are computed strictly causally: rolling windows and lags use
    only data available at each bar's timestamp.

    Args:
        btc_df:        BTC DataFrame from :func:`~polymarket.settlement.load_btc_hourly`.
                       Should include all history needed for rolling features
                       (at least 7 days before the earliest bar of interest).
        K:             Strike price in USD.
        T:             Settlement timestamp (UTC-aware Timestamp).
        model_package: Dict from :func:`load_model` or :func:`train`.

    Returns:
        DataFrame with columns ``['dt_utc', 'fair_value', 'log_moneyness', 'sigma_h']``.
        ``fair_value`` is the calibrated P(BTC_T > K) ∈ (0, 1).
        ``log_moneyness`` and ``sigma_h`` are included so that
        :class:`~alpha.EventProbAlpha` can apply the OTM distance penalty.
        Indexed by integer position (set ``dt_utc`` as index if needed).
    """
    T = T if T.tzinfo is not None else T.tz_localize("UTC")
    model = model_package["model"]
    feature_cols = model_package["feature_cols"]

    # Compute all features on the full BTC series (causal rolling windows are safe)
    df = add_btc_features(btc_df)
    df = add_contract_features(df, K, T)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Only predict for bars strictly before settlement
    df = df[df["dt"] < T].copy()
    df = df.dropna(subset=feature_cols)

    if df.empty:
        logger.warning(
            "predict_contract_probs: no valid rows for K=%s T=%s after feature engineering",
            K, T,
        )
        return pd.DataFrame(columns=["dt_utc", "fair_value"])

    X = df[feature_cols].values.astype(float)
    proba = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "dt_utc":        df["dt"].tolist(),  # preserve tz-awareness (.values strips it)
        "fair_value":    proba,
        "log_moneyness": df["log_moneyness"].values,
        "sigma_h":       df["sigma_h"].values,
    })

"""Orchestrator: contract metadata → settlements → synthetic feather files → predictions.

Main entry points (called by ``scripts/prepare_event_model.py``):

* :func:`build_all_feathers`        — writes synthetic OHLCV feather files for each
  contract (step 0 of data preparation).
* :func:`build_event_training_data` — builds the training dataset for the event model.
* :func:`build_event_predictions`   — generates per-contract fair-value CSVs that the
  strategy alpha factor reads at runtime.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from polymarket.contracts import ContractMetadata, load_contracts
from polymarket.settlement import load_btc_hourly, verify_settlements
from polymarket.synthetic_prices import _calibrate_sigma, build_synthetic_ohlcv


def _pair_to_filename(pair: str, timeframe: str = "1h") -> str:
    """Convert a freqtrade pair string to a feather file name.

    ``"BTCABOVE90K-JAN20-YES/USDT"``  →  ``"BTCABOVE90K-JAN20-YES_USDT-1h.feather"``
    """
    return pair.replace("/", "_") + f"-{timeframe}.feather"


def build_all_feathers(
    jsonl_path: str | Path,
    btc_csv_path: str | Path,
    output_dir: str | Path,
    strikes_filter: list[float] | None = None,
    yes_only: bool = True,
    noise_std: float = 0.015,
    random_seed: int = 42,
    calibration_months: int = 6,
    timeframe: str = "1h",
    verify: bool = True,
) -> list[ContractMetadata]:
    """Build synthetic OHLCV feather files for all (or selected) contracts.

    Args:
        jsonl_path:          Path to the ``.jsonl`` contract metadata file.
        btc_csv_path:        Path to ``data_1h.csv``.
        output_dir:          Directory to write ``.feather`` files into.
        strikes_filter:      If provided, only contracts whose strike is in this
                             list are processed.  ``None`` processes all contracts.
        yes_only:            If ``True``, only write YES-side feathers.
                             Set to ``False`` to also write NO-side feathers
                             (complement prices).
        noise_std:           Gaussian noise std dev for synthetic prices.
        random_seed:         Base random seed (each contract gets seed + index).
        calibration_months:  Months of BTC history used for sigma calibration.
        timeframe:           Timeframe label embedded in the file name.
        verify:              If ``True``, cross-check settlements against BTC data
                             and print a summary table.

    Returns:
        List of :class:`~polymarket.contracts.ContractMetadata` for the
        contracts that were actually written.

    Raises:
        ValueError: If any settlement cross-check fails.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load contracts and BTC data
    contracts = load_contracts(jsonl_path)
    btc_df = load_btc_hourly(str(btc_csv_path))

    # Optional: cross-check settlements
    if verify:
        rows = verify_settlements(contracts, btc_df)
        print(f"\n{'Slug':<50} {'Strike':>8} {'ResPrice':>10} {'OutcPrices':>11} {'BtcDerived':>11} {'OK':>4}")
        print("-" * 100)
        mismatches = 0
        for r in rows:
            ok = "YES" if r["match"] else "NO "
            if not r["match"]:
                mismatches += 1
            print(
                f"{r['slug']:<50} {r['strike']:>8,.0f} {r['resolution_price']:>10,.1f} "
                f"{r['outcome_prices_settlement']:>11.3f} {r['btc_derived_settlement']:>11.3f} {ok:>4}"
            )
        if mismatches:
            raise ValueError(
                f"{mismatches} settlement(s) do not match between outcomePrices and BTC data.  "
                "Check the resolution price logic in polymarket/settlement.py."
            )
        print(f"\nAll {len(rows)} settlements verified.\n")

    # Filter by strikes if requested
    selected = contracts
    if strikes_filter is not None:
        strike_set = set(strikes_filter)
        selected = [c for c in contracts if c.strike in strike_set]
        if not selected:
            raise ValueError(f"No contracts matched strikes_filter={strikes_filter}")

    # Calibrate sigma once (shared across all contracts from same start period)
    start_ts = pd.Timestamp(selected[0].start_date_utc)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    start_ts = start_ts.ceil("h")
    sigma_1h = _calibrate_sigma(btc_df, start_ts, months=calibration_months)
    print(f"Calibrated 1h BTC σ = {sigma_1h:.5f} ({sigma_1h * 100:.3f}%)  "
          f"(window: {calibration_months} months before {start_ts.date()})\n")

    written: list[ContractMetadata] = []

    for idx, contract in enumerate(selected):
        seed = random_seed + idx

        sides = ["yes"]
        if not yes_only:
            sides.append("no")

        for side in sides:
            pair = contract.pair_yes if side == "yes" else contract.pair_no
            filename = _pair_to_filename(pair, timeframe)
            filepath = output_path / filename

            ohlcv_df = build_synthetic_ohlcv(
                btc_df=btc_df,
                contract=contract,
                sigma_1h=sigma_1h,
                noise_std=noise_std,
                random_seed=seed,
            )

            # For NO side: complement prices (close = 1 - yes_close), re-clamp
            if side == "no":
                from polymarket.synthetic_prices import PRICE_CEIL, PRICE_FLOOR
                ohlcv_df = ohlcv_df.copy()
                ohlcv_df["close"] = (1.0 - ohlcv_df["close"]).clip(PRICE_FLOOR, PRICE_CEIL)
                ohlcv_df["open"] = (1.0 - ohlcv_df["open"]).clip(PRICE_FLOOR, PRICE_CEIL)
                ohlcv_df["high"] = (1.0 - ohlcv_df["low"]).clip(PRICE_FLOOR, PRICE_CEIL)
                ohlcv_df["low"] = (1.0 - ohlcv_df["high"]).clip(PRICE_FLOOR, PRICE_CEIL)
                # Patch settlement: NO settlement = 1.0 - YES settlement
                no_settlement = 1.0 - contract.settlement
                from polymarket.synthetic_prices import PRICE_CEIL as PC
                from polymarket.synthetic_prices import PRICE_FLOOR as PF
                ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = PC if no_settlement == 1.0 else PF

            table = pa.Table.from_pandas(ohlcv_df, preserve_index=False)
            feather.write_feather(table, str(filepath))
            print(f"  Wrote {filepath.name}  ({len(ohlcv_df)} candles, "
                  f"settlement={contract.settlement:.0f}, "
                  f"open={ohlcv_df['close'].iloc[0]:.3f}, "
                  f"final={ohlcv_df['close'].iloc[-1]:.3f})")

        written.append(contract)

    print(f"\nDone. Wrote feathers for {len(written)} contracts into {output_path}/")
    return written


def build_event_training_data(
    btc_csv_path: str | Path,
    output_path: str | Path,
    start_date: str = "2018-01-01",
    end_date: str = "2025-06-01",
    window_days: int = 7,
    day_of_week: int = 0,
    hour_utc: int = 17,
    relative_strikes: list[float] | None = None,
) -> pd.DataFrame:
    """Build and save the training dataset for the event-probability model.

    Constructs synthetic weekly BTC binary-event samples from ``btc_csv_path``
    and writes them as a Parquet file at ``output_path``.

    Args:
        btc_csv_path:     Path to ``data_1h.csv``.
        output_path:      Where to write the Parquet training data.
        start_date:       Earliest settlement date to include.
        end_date:         Latest settlement date (exclusive).
        window_days:      Contract duration in days (default 7).
        day_of_week:      Settlement weekday: 0=Mon … 6=Sun.
        hour_utc:         Settlement hour UTC.
        relative_strikes: Strike multipliers (default [0.85 … 1.15]).

    Returns:
        The constructed samples DataFrame (also saved to ``output_path``).
    """
    from polymarket.event_dataset import build_training_samples

    btc_df = load_btc_hourly(str(btc_csv_path))
    samples = build_training_samples(
        btc_df,
        start_date=start_date,
        end_date=end_date,
        window_days=window_days,
        day_of_week=day_of_week,
        hour_utc=hour_utc,
        relative_strikes=relative_strikes,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(str(out), index=False)
    print(
        f"Wrote {len(samples):,} training samples "
        f"({samples['T'].nunique()} events, "
        f"{samples['label'].mean():.3f} base rate) → {out}"
    )
    return samples


def train_event_model(
    training_data_path: str | Path,
    output_model_path: str | Path,
    val_cutoff: str = "2024-01-01",
    model_type: str = "logistic",
) -> dict:
    """Train the event-probability model and save it to disk.

    Loads the Parquet dataset produced by :func:`build_event_training_data`,
    trains a calibrated classifier, prints evaluation metrics, and writes
    the model package as a joblib pickle.

    Args:
        training_data_path: Path to the Parquet training data.
        output_model_path:  Where to write the model pickle.
        val_cutoff:         ISO date string for train/val split.
        model_type:         ``'logistic'`` (default) or ``'xgboost'``.

    Returns:
        The trained model package dict.
    """
    import pandas as pd

    from polymarket.event_model import save_model, train

    samples = pd.read_parquet(str(training_data_path))
    # Ensure T is UTC-aware after parquet round-trip
    if samples["T"].dt.tz is None:
        samples["T"] = samples["T"].dt.tz_localize("UTC")

    print(f"Loaded {len(samples):,} training samples from {training_data_path}")

    model_package = train(samples, val_cutoff=val_cutoff, model_type=model_type)
    save_model(model_package, output_model_path)

    print("\nModel metrics:")
    for split, m in model_package["metrics"].items():
        if m:
            print(
                f"  {split:<6}  AUC={m['auc']:.4f}  "
                f"Brier={m['brier']:.4f}  Acc={m['accuracy']:.4f}  n={m['n']:,}"
            )
    return model_package


def build_event_predictions(
    btc_csv_path: str | Path,
    model_path: str | Path,
    contracts: list[ContractMetadata],
    output_dir: str | Path,
) -> None:
    """Generate per-contract event probability CSVs for backtesting.

    For each contract, runs the event-probability model at every hourly bar
    in ``btc_csv_path`` before the contract's settlement time and writes a
    CSV named ``{pair_yes_filename}-event_probs.csv``.

    The output CSVs are read at strategy runtime by
    :class:`~user_data.strategies.DualModelPolymarketPortfolio` via
    :class:`~alpha.EventProbAlpha`.

    Args:
        btc_csv_path: Path to ``data_1h.csv``.
        model_path:   Path to the saved event model pickle.
        contracts:    List of :class:`~polymarket.contracts.ContractMetadata`.
        output_dir:   Directory to write the per-contract CSVs.
    """
    from polymarket.event_model import load_model, predict_contract_probs

    btc_df = load_btc_hourly(str(btc_csv_path))
    model_package = load_model(model_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for contract in contracts:
        T = pd.Timestamp(contract.end_date_utc, tz="UTC")
        K = contract.strike
        pair = contract.pair_yes

        filename = pair.replace("/", "_") + "-event_probs.csv"
        filepath = out_dir / filename

        probs_df = predict_contract_probs(btc_df, K, T, model_package)

        if probs_df.empty:
            print(f"  WARNING: no predictions for {pair} — skipping")
            continue

        probs_df.to_csv(str(filepath), index=False)
        print(
            f"  Wrote {filepath.name}  "
            f"({len(probs_df)} bars, "
            f"fair_value: {probs_df['fair_value'].mean():.3f} mean, "
            f"settlement={contract.settlement:.0f})"
        )

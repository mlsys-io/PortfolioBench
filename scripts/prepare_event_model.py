"""End-to-end preparation script for the direct event-probability model.

Run this script once before backtesting with DualModelPolymarketPortfolio.
It performs four steps:

  0. Build feather files  — OHLCV price series per contract (synthetic by default;
                            use ``--use-real-data`` for real Polymarket prices).
  1. Build training data  — synthetic weekly BTC events from data_1h.csv.
  2. Train event model    — calibrated logistic regression, saved to pkl.
  3. Generate predictions — per-contract fair_value CSVs for the backtester.

Usage
-----
From the repo root::

    # Default: synthetic OHLCV from a JSONL contracts file
    python scripts/prepare_event_model.py

    # Real Polymarket prices from a trade-history parquet
    python scripts/prepare_event_model.py \\
        --use-real-data \\
        --parquet-path mycode/data/combined_filtered_data.paquet \\
        --output-dir   user_data/data/polymarket_ml_real

All paths default to the standard repo layout.  Override via CLI flags::

    python scripts/prepare_event_model.py \\
        --btc-csv      mycode/data/data_1h.csv \\
        --contracts    user_data/data/polymarket_contracts/jan20.jsonl \\
        --output-dir   user_data/data/polymarket_ml \\
        --start-date   2018-01-01 \\
        --end-date     2025-06-01 \\
        --val-cutoff   2024-01-01 \\
        --model-type   logistic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on the Python path when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_event_model")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--btc-csv",     default="mycode/data/data_1h.csv")
    p.add_argument("--contracts",   default="user_data/data/polymarket_contracts/jan20.jsonl")
    p.add_argument("--output-dir",  default="user_data/data/polymarket_ml")
    p.add_argument("--start-date",  default="2018-01-01")
    p.add_argument("--end-date",    default="2025-06-01")
    p.add_argument("--val-cutoff",  default="2024-01-01",
                   help="Events before this date go to training.")
    p.add_argument("--model-type",  default="logistic", choices=["logistic", "xgboost"])
    p.add_argument("--skip-feathers", action="store_true",
                   help="Skip step 0 if feather files already exist in output-dir.")
    p.add_argument("--skip-training-data", action="store_true",
                   help="Skip step 1 if training_data.parquet already exists.")
    # Real-data mode
    p.add_argument("--use-real-data", action="store_true",
                   help="Step 0: build feathers from real Polymarket trade data "
                        "instead of synthetic prices. Requires --parquet-path.")
    p.add_argument("--parquet-path", default="mycode/data/combined_filtered_data.paquet",
                   help="Path to the Polymarket trade-history parquet file "
                        "(used only with --use-real-data).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    btc_csv      = REPO_ROOT / args.btc_csv
    contracts_path = REPO_ROOT / args.contracts
    output_dir   = REPO_ROOT / args.output_dir

    training_data_path = output_dir / "event_model_training.parquet"
    model_path         = output_dir / "event_model.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0: Build feather files (synthetic or real)
    # ------------------------------------------------------------------
    if args.skip_feathers:
        logger.info("Step 0 skipped — assuming feather files already exist in %s", output_dir)
    elif args.use_real_data:
        logger.info("Step 0/3 — Building real-data OHLCV feather files from parquet")
        parquet_path = REPO_ROOT / args.parquet_path
        from polymarket.real_data_builder import build_all_feathers_from_parquet
        written = build_all_feathers_from_parquet(
            parquet_path=parquet_path,
            output_dir=output_dir,
        )
        logger.info("Step 0 complete: %d feather files written", len(written))
        # In real-data mode, the contracts come from the parquet itself; skip
        # the JSONL-based contracts file for step 3 (we'll derive them below).
        contracts_path = None  # signal to step 3 to use the parsed list
        _real_contracts = written
    else:
        logger.info("Step 0/3 — Building synthetic OHLCV feather files")
        from polymarket.data_builder import build_all_feathers
        build_all_feathers(
            jsonl_path=contracts_path,
            btc_csv_path=btc_csv,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Step 1: Build training data
    # ------------------------------------------------------------------
    if args.skip_training_data and training_data_path.exists():
        logger.info("Step 1 skipped — using existing %s", training_data_path)
    else:
        logger.info("Step 1/3 — Building training data (%s → %s)", btc_csv, training_data_path)
        from polymarket.data_builder import build_event_training_data
        build_event_training_data(
            btc_csv_path=btc_csv,
            output_path=training_data_path,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    # ------------------------------------------------------------------
    # Step 2: Train event model
    # ------------------------------------------------------------------
    logger.info("Step 2/3 — Training event model (val_cutoff=%s, type=%s)",
                args.val_cutoff, args.model_type)
    from polymarket.data_builder import train_event_model
    train_event_model(
        training_data_path=training_data_path,
        output_model_path=model_path,
        val_cutoff=args.val_cutoff,
        model_type=args.model_type,
    )

    # ------------------------------------------------------------------
    # Step 3: Generate per-contract predictions
    # ------------------------------------------------------------------
    logger.info("Step 3/3 — Generating per-contract predictions")
    from polymarket.data_builder import build_event_predictions

    if args.use_real_data and not args.skip_feathers:
        # Contracts were parsed from the parquet in step 0; JSONL also written
        contracts = _real_contracts
        logger.info("  Using %d contracts parsed from parquet", len(contracts))
    elif args.use_real_data and args.skip_feathers:
        # Feathers already built — reload contracts from the written JSONL
        from polymarket.contracts import load_contracts as _load
        real_jsonl = output_dir / "real_contracts.jsonl"
        contracts = _load(real_jsonl)
        logger.info("  Loaded %d contracts from %s", len(contracts), real_jsonl)
    else:
        from polymarket.contracts import load_contracts
        contracts = load_contracts(contracts_path)
        logger.info("  Loaded %d contracts from %s", len(contracts), contracts_path)

    build_event_predictions(
        btc_csv_path=btc_csv,
        model_path=model_path,
        contracts=contracts,
        output_dir=output_dir,
    )

    logger.info("Done.  Run the backtest with:")
    logger.info(
        "  portbench backtesting "
        "--strategy DualModelPolymarketPortfolio "
        "--strategy-path ./user_data/strategies "
        "--config user_data/config_polymarket_ml.json "
        "--datadir user_data/data/polymarket_ml "
        "--timerange 20260113-20260121"
    )


if __name__ == "__main__":
    main()

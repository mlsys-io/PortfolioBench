"""
Orchestrates the full Polymarket dataset pipeline in order:

  1. parser.py                      — parse JSONL → markets/tokens/filtered_token_ids parquet
  2. fetch_orderbook.py             — fetch raw orderbook snapshots per token
  3. orderbook_feature_generation.py — build per-token feature parquet files

Run with defaults:
    python dataset/polymarket_orderbook/run_pipeline.py

Skip a stage you've already completed:
    python dataset/polymarket_orderbook/run_pipeline.py --skip-parse --skip-fetch
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"STAGE: {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: stage '{label}' failed (exit {result.returncode}). Aborting.")
        sys.exit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full Polymarket dataset pipeline end-to-end."
    )

    # Stage skips
    parser.add_argument("--skip-parse", action="store_true", help="Skip parser.py")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch_orderbook.py")
    parser.add_argument("--skip-features", action="store_true", help="Skip orderbook_feature_generation.py")

    # Parser args
    parser.add_argument("--market-jsonl", type=str, help="Path to polymarket_markets_1y.jsonl")
    parser.add_argument("--min-end-date", type=str, default="2026-01-01")
    parser.add_argument("--top-markets", type=int, default=70)

    # Fetch args
    parser.add_argument("--start-date", type=str, default="2025-10-14")
    parser.add_argument("--end-date", type=str, default="2026-03-31")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Re-fetch already-saved orderbook files")

    # Feature pipeline args
    parser.add_argument("--depth-n", type=int, default=3)
    parser.add_argument("--process-all-files", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Stage 1: Parse ---
    if not args.skip_parse:
        cmd = [PYTHON, str(DATASET_DIR / "parser.py"),
               "--min-end-date", args.min_end_date,
               "--top-markets", str(args.top_markets)]
        if args.market_jsonl:
            cmd += ["--market-jsonl", args.market_jsonl]
        run(cmd, "parser.py")

    # --- Stage 2: Fetch orderbooks ---
    if not args.skip_fetch:
        cmd = [PYTHON, str(DATASET_DIR / "fetch_orderbook.py"),
               "--start-date", args.start_date,
               "--end-date", args.end_date,
               "--max-concurrent", str(args.max_concurrent),
               "--batch-size", str(args.batch_size)]
        if args.force:
            cmd.append("--force")
        run(cmd, "fetch_orderbook.py")

    # --- Stage 3: Build features ---
    if not args.skip_features:
        cmd = [PYTHON, str(DATASET_DIR / "orderbook_feature_generation.py"),
               "--depth-n", str(args.depth_n)]
        if args.process_all_files:
            cmd.append("--process-all-files")
        run(cmd, "orderbook_feature_generation.py")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

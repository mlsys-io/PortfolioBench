import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.polymarket_orderbook.utils.paths import FILTERED_TOKEN_IDS, ORDERBOOK_DIR, ensure_dirs
from dataset.polymarket_orderbook.utils.fetch_orderbook import fetch_orderbook_from_ids_async


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch Polymarket orderbook snapshots from DomeAPI for filtered token IDs."
    )
    parser.add_argument("--filtered-tokens-path", type=Path, default=FILTERED_TOKEN_IDS)
    parser.add_argument("--raw-orderbook-dir", type=Path, default=ORDERBOOK_DIR)
    parser.add_argument("--start-date", type=str, default="2025-10-14")
    parser.add_argument("--end-date", type=str, default="2026-03-31")
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of tokens to process per batch before freeing memory (default: 10).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch tokens that already have a saved parquet file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    args.raw_orderbook_dir.mkdir(parents=True, exist_ok=True)

    # Load token IDs from the filtered parquet produced by polymarket_parser.py
    token_ids_df = pd.read_parquet(args.filtered_tokens_path)
    all_token_ids = token_ids_df["token_id"].astype(str).tolist()
    print(f"Loaded {len(all_token_ids)} token IDs from {args.filtered_tokens_path}")

    # Resume: skip tokens whose output file already exists
    if args.force:
        token_ids = all_token_ids
    else:
        already_done = {
            p.stem[len("ob_"):] for p in args.raw_orderbook_dir.glob("ob_*.parquet")
        }
        token_ids = [t for t in all_token_ids if t not in already_done]
        if already_done:
            print(
                f"Skipping {len(already_done)} already-fetched token(s). "
                f"{len(token_ids)} remaining. Use --force to re-fetch all."
            )

    if not token_ids:
        print("Nothing to fetch.")
        return

    print(f"Fetching orderbook for {len(token_ids)} token(s) ({args.start_date} → {args.end_date})")

    fetch_orderbook_from_ids_async(
        token_ids=token_ids,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.raw_orderbook_dir,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

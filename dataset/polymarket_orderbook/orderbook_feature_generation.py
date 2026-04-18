import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset.polymarket_orderbook.utils.paths import (
    DATA_DIR,
    FILTERED_TOKEN_IDS,
    ORDERBOOK_DIR,
    FEATURE_DIR,
    ensure_dirs,
)
from dataset.polymarket_orderbook.utils.orderbook_feature_generation import build_token_feature_table_from_parquet


def parse_args():
    parser = argparse.ArgumentParser(description="Build per-token orderbook feature parquet files.")
    parser.add_argument("--tokens-path", type=Path, default=DATA_DIR / "tokens.parquet")
    parser.add_argument("--filtered-tokens-path", type=Path, default=FILTERED_TOKEN_IDS)
    parser.add_argument("--raw-orderbook-dir", type=Path, default=ORDERBOOK_DIR)
    parser.add_argument("--feat-orderbook-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--depth-n", type=int, default=3)
    parser.add_argument("--drop-json-cols", action="store_true", default=True)
    parser.add_argument(
        "--process-all-files",
        action="store_true",
        help="Ignore filtered token list and process all ob_*.parquet files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    args.feat_orderbook_dir.mkdir(parents=True, exist_ok=True)

    # Load filtered token IDs
    relevant_token_ids = set(pd.read_parquet(args.filtered_tokens_path)["token_id"].astype(str))
    print(f"Loaded {len(relevant_token_ids)} filtered token IDs")

    # Load token metadata, restricted to filtered set
    tokens_df = pd.read_parquet(args.tokens_path)
    tokens_df = tokens_df[tokens_df["token_id"].astype(str).isin(relevant_token_ids)]
    print(f"Token metadata rows after filter: {len(tokens_df):,}")

    files = sorted(args.raw_orderbook_dir.glob("ob_*.parquet"))
    if not files:
        print(f"No ob_*.parquet files found in {args.raw_orderbook_dir}")
        return

    if not args.process_all_files:
        files = [p for p in files if p.stem[len("ob_"):] in relevant_token_ids]

    print(f"Files to process: {len(files)}")

    success = 0
    failed = 0

    for i, file_path in enumerate(files, start=1):
        token_stub = file_path.stem[len("ob_"):]
        output_path = args.feat_orderbook_dir / f"feat_{token_stub}.parquet"

        print(f"[{i}/{len(files)}] {file_path.name}")
        try:
            feat_df = build_token_feature_table_from_parquet(
                input_path=file_path,
                output_path=output_path,
                depth_n=args.depth_n,
                drop_json_cols=args.drop_json_cols,
                token_meta_df=tokens_df,
            )
            print(f"  Saved {len(feat_df):,} rows -> {output_path.name}")
            success += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1

    print(f"Done. Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()

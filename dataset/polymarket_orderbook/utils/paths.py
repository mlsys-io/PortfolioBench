from pathlib import Path

# Project root = PortfolioBench/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "user_data" / "data" / "polymarket"
ORDERBOOK_DIR = DATA_DIR / "raw_orderbook"
FEATURE_DIR = DATA_DIR / "feat_orderbook"

MARKET_JSONL = DATA_DIR / "polymarket_markets_1y.jsonl"

# User-defined token filter output/input shared by both notebooks
FILTERED_TOKEN_IDS = DATA_DIR / "filtered_token_ids.parquet"

def ensure_dirs() -> None:
    for p in [DATA_DIR, ORDERBOOK_DIR, FEATURE_DIR]:
        p.mkdir(parents=True, exist_ok=True)
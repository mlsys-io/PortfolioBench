#!/usr/bin/env python3
"""Download historical event contract data from Polymarket's CLOB API.

Fetches OHLCV-style candle data for Polymarket binary outcome contracts
and stores them as feather files compatible with the PortfolioBench data
pipeline.

Polymarket CLOB API docs: https://docs.polymarket.com/

Usage:
    python utils/download_polymarket_data.py
    python utils/download_polymarket_data.py --events "will-trump-win-2024,will-eth-hit-10k"
    python utils/download_polymarket_data.py --slug will-trump-win-2024 --timeframe 1d
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Polymarket API endpoints
# ---------------------------------------------------------------------------
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# Timeframe mapping: PortfolioBench timeframe -> Polymarket interval seconds
TIMEFRAME_MAP = {
    "5m": 300,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Default output directory
DEFAULT_DATA_DIR = Path("user_data/data/polymarket")


def fetch_events(limit: int = 50, active: bool = True) -> list[dict]:
    """Fetch event list from the Gamma API."""
    if requests is None:
        raise ImportError("requests library required: pip install requests")

    params = {"limit": limit, "active": str(active).lower(), "order": "volume"}
    resp = requests.get(f"{GAMMA_BASE_URL}/events", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_market_candles(
    token_id: str,
    interval: int = 86400,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    """Fetch price candle history for a specific contract token from the CLOB API."""
    if requests is None:
        raise ImportError("requests library required: pip install requests")

    params: dict = {"market": token_id, "interval": str(interval)}
    if start_ts:
        params["startTs"] = str(start_ts)
    if end_ts:
        params["endTs"] = str(end_ts)

    all_candles: list[dict] = []
    fid = None

    while True:
        if fid:
            params["fid"] = fid
        resp = requests.get(f"{CLOB_BASE_URL}/prices-history", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        history = data if isinstance(data, list) else data.get("history", [])
        if not history:
            break
        all_candles.extend(history)
        next_cursor = data.get("next_cursor") if isinstance(data, dict) else None
        if not next_cursor:
            break
        fid = next_cursor
        time.sleep(0.5)  # rate-limit courtesy

    return all_candles


def candles_to_dataframe(candles: list[dict]) -> pd.DataFrame:
    """Convert Polymarket candle records to a standard OHLCV DataFrame.

    Polymarket candles have fields: t (timestamp), o, h, l, c, v
    Prices represent probability [0, 1].
    """
    if not candles:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    rows = []
    for c in candles:
        rows.append(
            {
                "date": int(float(c.get("t", c.get("timestamp", 0)))) * 1000,  # ms
                "open": float(c.get("o", c.get("open", 0))),
                "high": float(c.get("h", c.get("high", 0))),
                "low": float(c.get("l", c.get("low", 0))),
                "close": float(c.get("c", c.get("close", 0))),
                "volume": float(c.get("v", c.get("volume", 0))),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def save_feather(df: pd.DataFrame, pair_slug: str, timeframe: str, data_dir: Path) -> Path:
    """Save DataFrame as feather file with PortfolioBench naming convention.

    Convention: {SLUG}-{YES|NO}_USDT-{timeframe}.feather
    """
    filename = f"{pair_slug.replace('/', '_')}-{timeframe}.feather"
    filepath = data_dir / filename
    df.to_feather(filepath, compression_level=9, compression="lz4")
    logger.info("Saved %s (%d rows)", filepath, len(df))
    return filepath


def download_event(
    slug: str,
    timeframe: str = "1d",
    data_dir: Path = DEFAULT_DATA_DIR,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[Path]:
    """Download candle data for both YES and NO contracts of an event.

    Returns list of saved file paths.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    interval = TIMEFRAME_MAP.get(timeframe, 86400)
    saved: list[Path] = []

    # Fetch event details to get token IDs
    try:
        resp = requests.get(f"{GAMMA_BASE_URL}/events", params={"slug": slug}, timeout=30)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            logger.warning("No event found for slug: %s", slug)
            return saved
        event = events[0] if isinstance(events, list) else events
    except Exception as e:
        logger.error("Failed to fetch event %s: %s", slug, e)
        return saved

    markets = event.get("markets", [])
    if not markets:
        logger.warning("No markets found for event: %s", slug)
        return saved

    for market in markets:
        outcome = market.get("outcome", "YES").upper()
        token_id = market.get("clobTokenIds", [None])
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if not token_id:
            continue

        pair_slug = f"{slug.upper()}-{outcome}_USDT"
        logger.info("Fetching candles for %s (token: %s)", pair_slug, token_id[:16])

        try:
            candles = fetch_market_candles(token_id, interval, start_ts, end_ts)
            df = candles_to_dataframe(candles)
            if not df.empty:
                path = save_feather(df, pair_slug, timeframe, data_dir)
                saved.append(path)
            else:
                logger.warning("No candle data for %s", pair_slug)
        except Exception as e:
            logger.error("Failed to fetch candles for %s: %s", pair_slug, e)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Download Polymarket event contract data")
    parser.add_argument(
        "--events",
        type=str,
        default="",
        help="Comma-separated event slugs to download",
    )
    parser.add_argument("--slug", type=str, help="Single event slug to download")
    parser.add_argument("--timeframe", type=str, default="1d", choices=list(TIMEFRAME_MAP.keys()))
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Output directory for feather files",
    )
    parser.add_argument("--list-events", action="store_true", help="List top active events")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.list_events:
        events = fetch_events(limit=20)
        for e in events:
            title = e.get("title", "Unknown")
            slug_val = e.get("slug", "")
            n_markets = len(e.get("markets", []))
            print(f"  {slug_val:40s}  ({n_markets} markets)  {title}")
        return

    slugs = []
    if args.slug:
        slugs = [args.slug]
    elif args.events:
        slugs = [s.strip() for s in args.events.split(",") if s.strip()]
    else:
        logger.info("No events specified. Use --slug or --events, or --list-events to browse.")
        return

    for slug in slugs:
        logger.info("=" * 60)
        logger.info("Downloading event: %s (timeframe: %s)", slug, args.timeframe)
        logger.info("=" * 60)
        download_event(slug, args.timeframe, data_dir)


if __name__ == "__main__":
    main()

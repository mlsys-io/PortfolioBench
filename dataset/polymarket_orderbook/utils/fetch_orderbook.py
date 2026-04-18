import json
import random
import time
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = "beb79777a5762ef81b41fbaae1dbb75d23fcee28"
DOME_HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "x-api-key": API_KEY,
    "Accept-Encoding": "identity",
}

DOME_URL = "https://api.domeapi.io/v1/polymarket/orderbooks"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_orderbook_from_ids_async(
    token_ids,
    start_date,
    end_date,
    output_path,
    max_concurrent=3,
    max_retries=6,
    batch_size=10,
):
    """Fetch orderbook snapshots for a list of token IDs and save as parquet files.
    """
    output_path = Path(output_path)

    dome_history_start = pd.Timestamp("2025-10-14", tz="UTC")

    def _to_utc_timestamp(value):
        if value is None or pd.isna(value):
            return None
        ts = pd.Timestamp(value)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    start_ts = _to_utc_timestamp(start_date)
    end_ts = _to_utc_timestamp(end_date)

    if start_ts is None or end_ts is None:
        raise ValueError("start_date and end_date must be valid timestamps")

    if start_ts < dome_history_start:
        print(f"Warning: start_date {start_ts} is before Dome history start {dome_history_start}. Adjusting.")
        start_ts = dome_history_start

    start_ms = int(start_ts.value // 1_000_000)
    end_ms = int(end_ts.value // 1_000_000)

    def _fetch_all_snapshots(session, token_id):
        params = {
            "token_id": str(token_id),
            "start_time": int(start_ms),
            "end_time": int(end_ms),
            "limit": 200,
        }

        all_snaps = []

        while True:
            for attempt in range(1, max_retries + 1):
                try:
                    resp = session.get(DOME_URL, params=params, headers=DOME_HEADERS, timeout=60)
                    if resp.status_code in RETRYABLE_STATUS:
                        raise requests.HTTPError(f"retryable status={resp.status_code}", response=resp)
                    resp.raise_for_status()
                    payload = resp.json()
                    break
                except Exception as e:
                    if attempt == max_retries:
                        raise RuntimeError(f"Failed token_id={token_id} after {max_retries} retries: {e}")
                    backoff = min(30, (2 ** attempt) + random.random())
                    print(f"Retry {attempt}/{max_retries} token_id={token_id} in {backoff:.1f}s ({e})")
                    time.sleep(backoff)

            snaps = payload.get("snapshots", [])
            all_snaps.extend(snaps)

            pagination = payload.get("pagination", {}) or {}
            has_more = pagination.get("has_more", False)
            pagination_key = pagination.get("pagination_key") or pagination.get("paginationKey")

            if not has_more or not pagination_key:
                break

            params["pagination_key"] = pagination_key
            time.sleep(0.05)

        return all_snaps

    def _save_orderbook_of_token(i, token_id):
        # Each thread gets its own requests.Session for connection isolation
        with requests.Session() as session:
            print(f"Processing token_id={token_id}, {i}/{len(token_ids)}")
            try:
                snapshots = _fetch_all_snapshots(session, token_id)
            except Exception as e:
                print(f"Skipping token_id={token_id}: {e}")
                return 0

        if not snapshots:
            return 0

        snapshots = sorted(
            snapshots,
            key=lambda s: (s.get("timestamp", 0), s.get("indexedAt", 0))
        )

        out = []
        for snap in snapshots:
            snap_ts = pd.to_datetime(snap.get("timestamp"), unit="ms", utc=True, errors="coerce")
            indexed_ts = pd.to_datetime(snap.get("indexedAt"), unit="ms", utc=True, errors="coerce")

            bids = snap.get("bids", []) or []
            asks = snap.get("asks", []) or []

            out.append({
                "token_id": token_id,
                "snapshot_time": snap_ts,
                "snapshot_timestamp_ms": snap.get("timestamp"),
                "indexed_at_time": indexed_ts,
                "indexed_at_ms": snap.get("indexedAt"),
                "market_hash": snap.get("market"),
                "asset_id": snap.get("assetId"),
                "tick_size": float(snap["tickSize"]) if snap.get("tickSize") is not None else np.nan,
                "min_order_size": float(snap["minOrderSize"]) if snap.get("minOrderSize") is not None else np.nan,
                "orderbook_neg_risk": snap.get("negRisk"),
                "bids_json": json.dumps(bids),
                "asks_json": json.dumps(asks),
            })

        token_df = pd.DataFrame(out).sort_values(["snapshot_time"], kind="stable").reset_index(drop=True)
        token_df.to_parquet(output_path / f"ob_{token_id}.parquet", index=False)
        n = len(token_df)
        print(f"Saved {n:,} rows for token_id={token_id}")
        return n

    all_counts = []
    for batch_start in range(0, len(token_ids), batch_size):
        batch = list(enumerate(token_ids[batch_start : batch_start + batch_size], start=batch_start + 1))
        with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = {pool.submit(_save_orderbook_of_token, i, tid): tid for i, tid in batch}
            for fut in as_completed(futures):
                try:
                    all_counts.append(fut.result())
                except Exception as e:
                    print(f"Unexpected error for token_id={futures[fut]}: {e}")
                    all_counts.append(0)
        print(f"Batch done: {batch_start + len(batch)}/{len(token_ids)} tokens processed.")

    print(f"Done processing all tokens. Total rows: {sum(all_counts):,}")

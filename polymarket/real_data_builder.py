"""Build backtesting feather files from real Polymarket trade data.

This module replaces the synthetic log-normal price generator
(:mod:`polymarket.synthetic_prices`) with real hourly OHLCV data extracted
from a Polymarket trade-history parquet file.

The parquet file is expected to have the following columns:

    timestamp    – datetime (UTC-aware)
    condition_id – market identifier string
    side         – outcome label, e.g. ``"Yes"`` / ``"No"``
    open         – string-encoded float price
    high         – string-encoded float price
    low          – string-encoded float price
    close        – string-encoded float price
    volume       – int64 traded volume
    question     – human-readable market question

Main entry points
-----------------
parse_btc_contracts(df)
    Scan all rows for BTC binary markets and return a list of parsed
    :class:`~polymarket.contracts.ContractMetadata` objects.

build_feather_from_real_data(df, contract, output_dir)
    Extract the 7-day window before expiry for one contract, forward-fill
    gaps, and write a freqtrade-compatible feather file.

build_all_feathers_from_parquet(parquet_path, output_dir, ...)
    End-to-end convenience wrapper: load the parquet, parse contracts,
    build feathers for all qualifying markets.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import pandas as pd
import pyarrow.feather as feather

from polymarket.contracts import ContractMetadata, _make_pair

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Backtesting window: last N hours before expiry
WINDOW_HOURS: int = 168  # 7 days

# Polymarket resolves BTC contracts at noon Eastern = 17:00 UTC
RESOLUTION_HOUR_UTC: int = 17

# Forward-fill limit: gaps longer than this are left as NaN and the contract
# is flagged as "sparse" (but still usable if overall coverage ≥ MIN_COVERAGE)
MAX_FFILL_HOURS: int = 6

# Minimum fraction of hourly candles required in the window
MIN_COVERAGE: float = 0.60  # 60% = at least 101 of 168 hours

# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

# "above $88,000" / "above $88K" / "above $88k"
# Group 1: digits, Group 2: optional K/k suffix
_ABOVE_RE = re.compile(r"\babove\s+\$([0-9,]+(?:\.[0-9]+)?)\s*([kK])?(?:\b|$)", re.I)
# "below $84,000" / "less than $84K" / "dip to $84,000"
_BELOW_RE = re.compile(
    r"\b(?:below|less than|dip(?:s)?\s+to)\s+\$([0-9,]+(?:\.[0-9]+)?)\s*([kK])?(?:\b|$)", re.I
)
# "reach $150,000" / "reach $100k" / "hit $100k" / "exceed $90,000"
_REACH_RE = re.compile(
    r"\b(?:reach|hit|exceed|surpass)\s+\$([0-9,]+(?:\.[0-9]+)?)\s*([kK])?(?:\b|$)", re.I
)

# Full month names and 3-letter abbreviations
_MONTH_MAP: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?(?:[,\s]+(\d{4}))?",
    re.I,
)

_YEAR_RE = re.compile(r"\b(20\d{2})\b")


def _parse_strike_direction(question: str) -> tuple[float, str] | None:
    """Return (strike_usd, direction) or None if unparseable."""
    for pattern, direction in (
        (_ABOVE_RE, "above"),
        (_REACH_RE, "above"),   # "reach $X" treated as above (YES if BTC > X)
        (_BELOW_RE, "below"),
    ):
        m = pattern.search(question)
        if m:
            raw = m.group(1).replace(",", "")
            has_k = m.group(2) is not None  # True when K/k suffix was captured
            strike = float(raw) * (1_000 if has_k else 1)
            # Skip implausible BTC prices (< $1,000 without K suffix)
            if strike < 1_000:
                return None
            return strike, direction
    return None


def _parse_expiry(question: str, last_timestamp: pd.Timestamp) -> str | None:
    """Return ISO-8601 UTC expiry string or None if unparseable.

    Resolution time is always 17:00 UTC (Polymarket noon ET rule).
    If the year is absent from the question we infer it from last_timestamp.
    """
    m = _DATE_RE.search(question)
    if not m:
        return None

    month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
    month = _MONTH_MAP.get(month_str.lower())
    if month is None:
        return None
    day = int(day_str)

    if year_str:
        year = int(year_str)
        try:
            expiry = pd.Timestamp(year=year, month=month, day=day,
                                  hour=RESOLUTION_HOUR_UTC, tz="UTC")
        except ValueError:
            return None
        return expiry.strftime("%Y-%m-%dT%H:%M:%SZ")

    # No year in question — try years from last_timestamp.year down to -2.
    # Pick the latest year whose parsed date is ≤ last_timestamp + 3 days
    # (small slack for contracts whose last candle is just after expiry).
    slack = pd.Timedelta(days=3)
    expiry = None
    for y in range(last_timestamp.year, last_timestamp.year - 3, -1):
        try:
            candidate = pd.Timestamp(year=y, month=month, day=day,
                                     hour=RESOLUTION_HOUR_UTC, tz="UTC")
        except ValueError:
            continue
        if candidate <= last_timestamp + slack:
            expiry = candidate
            break

    if expiry is None:
        return None

    return expiry.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Parquet scanning
# ---------------------------------------------------------------------------

def parse_btc_contracts(
    df: pd.DataFrame,
    min_rows_last7d: int = int(WINDOW_HOURS * MIN_COVERAGE),
) -> list[ContractMetadata]:
    """Scan the parquet DataFrame and return parseable BTC binary contracts.

    Args:
        df:               Full parquet DataFrame (all columns).
        min_rows_last7d:  Minimum rows in the last 7-day window to include.

    Returns:
        List of :class:`~polymarket.contracts.ContractMetadata` objects, one
        per qualifying (condition_id, YES-side) market.
    """
    # Keep only BTC/Bitcoin YES-side rows
    btc_mask = (
        df["question"].str.contains("Bitcoin|BTC", case=False, na=False)
        & (df["side"].str.strip().str.lower() == "yes")
    )
    btc = df[btc_mask].copy()
    btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)

    contracts: list[ContractMetadata] = []
    skipped = {"no_strike": 0, "no_date": 0, "sparse": 0}

    for cid, grp in btc.groupby("condition_id"):
        question = grp["question"].iloc[0]
        last_ts = grp["timestamp"].max()

        # Parse strike and direction
        result = _parse_strike_direction(question)
        if result is None:
            skipped["no_strike"] += 1
            continue
        strike, direction = result

        # Parse expiry
        end_date_utc = _parse_expiry(question, last_ts)
        if end_date_utc is None:
            skipped["no_date"] += 1
            continue

        # Check coverage in the last 7-day window
        window_start = last_ts - pd.Timedelta(hours=WINDOW_HOURS)
        rows_in_window = (grp["timestamp"] >= window_start).sum()
        if rows_in_window < min_rows_last7d:
            skipped["sparse"] += 1
            continue

        # Determine settlement: final close ≥ 0.5 → YES won
        final_close = pd.to_numeric(grp["close"], errors="coerce").dropna()
        if final_close.empty:
            skipped["sparse"] += 1
            continue
        # Use the last candle's close as settlement proxy
        settlement = 1.0 if float(final_close.iloc[-1]) >= 0.5 else 0.0

        pair_yes = _make_pair(strike, direction, end_date_utc, "YES")
        pair_no = _make_pair(strike, direction, end_date_utc, "NO")

        contracts.append(
            ContractMetadata(
                id=str(cid),
                question=question,
                slug=str(cid),
                strike=strike,
                direction=direction,
                end_date_utc=end_date_utc,
                start_date_utc=(
                    grp["timestamp"].min().strftime("%Y-%m-%dT%H:%M:%SZ")
                ),
                settlement=settlement,
                volume_usd=float(grp["volume"].sum()),
                pair_yes=pair_yes,
                pair_no=pair_no,
                raw={"condition_id": cid, "rows_in_window": int(rows_in_window)},
            )
        )

    logger.info(
        "parse_btc_contracts: %d contracts parsed, skipped %d (no_strike=%d, "
        "no_date=%d, sparse=%d)",
        len(contracts), sum(skipped.values()),
        skipped["no_strike"], skipped["no_date"], skipped["sparse"],
    )
    return contracts


# ---------------------------------------------------------------------------
# Feather builder
# ---------------------------------------------------------------------------

def build_feather_from_real_data(
    df: pd.DataFrame,
    contract: ContractMetadata,
    output_dir: str | Path,
    *,
    max_ffill_hours: int = MAX_FFILL_HOURS,
) -> Path:
    """Extract real OHLCV for one contract and write a freqtrade feather file.

    The function:

    1. Filters the parquet to the contract's ``condition_id`` and ``side="Yes"``.
    2. Selects the last :data:`WINDOW_HOURS` hours before expiry.
    3. Converts string prices to float and timestamps to millisecond integers.
    4. Reindexes to a full hourly grid and forward-fills gaps up to
       ``max_ffill_hours`` consecutive missing candles.
    5. Writes ``{pair_yes}-1h.feather`` to ``output_dir``.

    Args:
        df:              Full parquet DataFrame.
        contract:        Parsed contract metadata.
        output_dir:      Directory to write the feather file.
        max_ffill_hours: Maximum consecutive hours to forward-fill.

    Returns:
        Path to the written feather file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cid = contract.raw.get("condition_id", contract.id)

    # Filter to this contract's YES side
    mask = (
        (df["condition_id"] == cid)
        & (df["side"].str.strip().str.lower() == "yes")
    )
    grp = df[mask].copy()
    if grp.empty:
        raise ValueError(f"No rows found for condition_id={cid!r}")

    grp["timestamp"] = pd.to_datetime(grp["timestamp"], utc=True)

    # 7-day window before expiry
    expiry_ts = pd.Timestamp(contract.end_date_utc.replace("Z", "+00:00"))
    window_start = expiry_ts - pd.Timedelta(hours=WINDOW_HOURS)
    grp = grp[(grp["timestamp"] >= window_start) & (grp["timestamp"] < expiry_ts)]

    if grp.empty:
        raise ValueError(
            f"No rows in 7-day window for {contract.pair_yes} "
            f"({window_start} – {expiry_ts})"
        )

    # Convert price strings to float
    for col in ("open", "high", "low", "close"):
        grp[col] = pd.to_numeric(grp[col], errors="coerce")

    grp = grp.set_index("timestamp").sort_index()

    # Reindex to full hourly grid
    full_index = pd.date_range(start=window_start, end=expiry_ts - pd.Timedelta(hours=1),
                               freq="h", tz="UTC")
    ohlcv = grp[["open", "high", "low", "close", "volume"]].reindex(full_index)

    # Forward-fill gaps up to max_ffill_hours
    ohlcv = ohlcv.ffill(limit=max_ffill_hours)

    # Log coverage
    n_filled = ohlcv["close"].notna().sum()
    coverage = n_filled / len(ohlcv)
    if coverage < MIN_COVERAGE:
        logger.warning(
            "%s: coverage %.0f%% is below %.0f%% threshold — feather written but "
            "results may be unreliable",
            contract.pair_yes, coverage * 100, MIN_COVERAGE * 100,
        )
    else:
        logger.info("%s: %.0f%% hourly coverage (%d/%d candles)",
                    contract.pair_yes, coverage * 100, n_filled, len(ohlcv))

    # Fill any remaining NaN with 0 for volume, and carry-forward for prices
    ohlcv["volume"] = ohlcv["volume"].fillna(0.0)
    ohlcv[["open", "high", "low", "close"]] = (
        ohlcv[["open", "high", "low", "close"]].ffill().bfill()
    )

    # Convert to freqtrade format: date as ms int64
    ohlcv = ohlcv.reset_index().rename(columns={"index": "date"})
    ohlcv["date"] = ohlcv["date"].astype("int64") // 1_000_000

    # Write feather (pair name → safe filename)
    pair_safe = contract.pair_yes.replace("/", "_")
    out_path = output_dir / f"{pair_safe}-1h.feather"
    feather.write_feather(ohlcv[["date", "open", "high", "low", "close", "volume"]], str(out_path))
    logger.info("Wrote %s (%d rows)", out_path, len(ohlcv))
    return out_path


# ---------------------------------------------------------------------------
# End-to-end convenience wrapper
# ---------------------------------------------------------------------------

def build_all_feathers_from_parquet(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    min_rows_last7d: int = int(WINDOW_HOURS * MIN_COVERAGE),
    max_ffill_hours: int = MAX_FFILL_HOURS,
    filter_condition_ids: Sequence[str] | None = None,
    write_jsonl: bool = True,
) -> list[ContractMetadata]:
    """Load the parquet, parse BTC contracts, and build feather files.

    Args:
        parquet_path:         Path to the Polymarket parquet file.
        output_dir:           Directory to write feather files.
        min_rows_last7d:      Minimum rows in last 7d window to include.
        max_ffill_hours:      Maximum gap fill length (hours).
        filter_condition_ids: If given, only process these condition IDs.

    Returns:
        List of :class:`~polymarket.contracts.ContractMetadata` for all
        contracts successfully written.
    """
    logger.info("Loading parquet from %s …", parquet_path)
    df = pd.read_parquet(
        str(parquet_path),
        columns=["timestamp", "condition_id", "side",
                 "open", "high", "low", "close", "volume", "question"],
    )

    if filter_condition_ids is not None:
        df = df[df["condition_id"].isin(filter_condition_ids)]

    contracts = parse_btc_contracts(df, min_rows_last7d=min_rows_last7d)
    logger.info("Building feathers for %d contracts …", len(contracts))

    written: list[ContractMetadata] = []
    for contract in contracts:
        try:
            build_feather_from_real_data(
                df, contract, output_dir, max_ffill_hours=max_ffill_hours
            )
            written.append(contract)
        except Exception as exc:
            logger.warning("Skipping %s: %s", contract.pair_yes, exc)

    if write_jsonl and written:
        jsonl_path = Path(output_dir) / "real_contracts.jsonl"
        write_contracts_jsonl(written, jsonl_path)

    logger.info(
        "Done. Wrote feathers for %d/%d contracts into %s",
        len(written), len(contracts), output_dir,
    )
    return written


# ---------------------------------------------------------------------------
# JSONL serialisation (for strategy contract registry)
# ---------------------------------------------------------------------------

def write_contracts_jsonl(
    contracts: list[ContractMetadata],
    output_path: str | Path,
) -> Path:
    """Serialise a list of :class:`~polymarket.contracts.ContractMetadata` to JSONL.

    The output format mirrors the Polymarket REST API schema expected by
    :func:`polymarket.contracts.load_contracts`, so the strategy can load
    real-data contracts through the same code path.

    Args:
        contracts:    Parsed contracts (e.g. from :func:`parse_btc_contracts`).
        output_path:  Path to write the ``.jsonl`` file.

    Returns:
        Path to the written file.
    """
    import json as _json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as fh:
        for c in contracts:
            # Encode settlement as Polymarket outcomePrices JSON string
            if c.settlement >= 0.5:
                outcome_prices = '["1.0", "0.0"]'
            else:
                outcome_prices = '["0.0", "1.0"]'

            record = {
                "id": c.id,
                "question": c.question,
                "slug": c.slug,
                "endDate": c.end_date_utc,
                "startDate": c.start_date_utc,
                "outcomePrices": outcome_prices,
                "volume": str(c.volume_usd),
            }
            fh.write(_json.dumps(record) + "\n")

    logger.info("Wrote %d contracts to %s", len(contracts), output_path)
    return output_path

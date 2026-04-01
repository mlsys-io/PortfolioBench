"""Contract metadata loading and normalisation.

The canonical metadata source is a JSONL file where each line is a Polymarket
market object as returned by the Polymarket REST API.  The Jan-20 batch lives at
``user_data/data/polymarket_contracts/jan20.jsonl``.

Each line is parsed into a :class:`ContractMetadata` dataclass, which adds:

* A normalised ``pair_yes`` / ``pair_no`` freqtrade pair string.
* The ``strike`` and ``direction`` extracted from the question text.
* The deterministic settlement value (0.0 or 1.0) derived from ``outcomePrices``.

The ``settlement`` field is computed from the ``outcomePrices`` JSON array that
Polymarket already stores on every resolved market.  We do **not** need the BTC
price for settlement at load time — that alternative path lives in
:mod:`polymarket.settlement` and can be used for double-checking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Pair naming helpers
# ---------------------------------------------------------------------------

_DIRECTION_TAG = {
    "above": "ABOVE",
    "below": "BELOW",
}

_MONTH_ABBR = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
    "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
    "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC",
}


def _make_pair(strike: float, direction: str, end_date_utc: str, side: str) -> str:
    """Build a freqtrade pair string for one side of a contract.

    Format: ``BTC{DIRECTION}{STRIKE_K}-{MON}{DAY}-{SIDE}/USDT``
    Example: ``BTCABOVE90K-JAN20-YES/USDT``

    Args:
        strike:        Strike price (e.g. 90000.0).
        direction:     ``'above'`` or ``'below'``.
        end_date_utc:  ISO-8601 string, e.g. ``"2026-01-20T17:00:00Z"``.
        side:          ``'YES'`` or ``'NO'``.
    """
    strike_k = int(strike // 1000)
    dir_tag = _DIRECTION_TAG[direction]
    month = _MONTH_ABBR[end_date_utc[5:7]]
    day = end_date_utc[8:10].lstrip("0") or "0"
    return f"BTC{dir_tag}{strike_k}K-{month}{day}-{side}/USDT"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContractMetadata:
    """Normalised representation of one Polymarket binary contract.

    Attributes
    ----------
    id:              Polymarket numeric market ID.
    question:        Human-readable question text.
    slug:            Polymarket URL slug.
    strike:          Strike price in USD.
    direction:       ``'above'`` or ``'below'``.
    end_date_utc:    Contract expiry as ISO-8601 UTC string.
    start_date_utc:  Contract open time as ISO-8601 UTC string.
    settlement:      1.0 if the YES outcome won, 0.0 if NO won.
    volume_usd:      Total traded volume at time of snapshot.
    pair_yes:        Freqtrade pair for the YES side.
    pair_no:         Freqtrade pair for the NO side.
    raw:             Original parsed dict (for debugging).
    """

    id: str
    question: str
    slug: str
    strike: float
    direction: str
    end_date_utc: str
    start_date_utc: str
    settlement: float          # 1.0 = YES won, 0.0 = NO won
    volume_usd: float
    pair_yes: str
    pair_no: str
    raw: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Strike / direction extraction
# ---------------------------------------------------------------------------

# Matches "above $90,000", "above $90k", "less than $84,000", etc.
_ABOVE_RE = re.compile(r"above\s+\$([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE)
_BELOW_RE = re.compile(r"(?:less than|below)\s+\$([0-9,]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _parse_strike_direction(question: str) -> tuple[float, str]:
    """Extract (strike, direction) from a Polymarket question string.

    Raises:
        ValueError: If neither 'above' nor 'below/less than' is found.
    """
    m = _ABOVE_RE.search(question)
    if m:
        strike = float(m.group(1).replace(",", ""))
        return strike, "above"

    m = _BELOW_RE.search(question)
    if m:
        strike = float(m.group(1).replace(",", ""))
        return strike, "below"

    raise ValueError(f"Cannot parse strike/direction from question: {question!r}")


# ---------------------------------------------------------------------------
# Settlement from outcomePrices
# ---------------------------------------------------------------------------

def _settlement_from_outcome_prices(outcome_prices_json: str) -> float:
    """Return 1.0 if the YES outcome won, else 0.0.

    Polymarket stores ``outcomePrices`` as a JSON-encoded list of strings.
    The list order is always ``[YES_price, NO_price]``.
    A resolved market will have one value at ``"1"`` and the other at ``"0"``.

    Examples::

        '["1", "0"]'  ->  1.0   (YES won)
        '["0", "1"]'  ->  0.0   (NO won)
    """
    prices = json.loads(outcome_prices_json)
    yes_price = float(prices[0])
    return yes_price  # already 0.0 or 1.0 for resolved markets


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_contracts(jsonl_path: str | Path) -> list[ContractMetadata]:
    """Parse a Polymarket JSONL snapshot into a list of :class:`ContractMetadata`.

    Args:
        jsonl_path: Path to the ``.jsonl`` file (one JSON object per line).

    Returns:
        List of :class:`ContractMetadata`, sorted by strike ascending.

    Raises:
        FileNotFoundError: If ``jsonl_path`` does not exist.
        ValueError:        If a line cannot be parsed.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Contract metadata file not found: {path}")

    contracts: list[ContractMetadata] = []

    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON parse error on line {lineno}: {exc}") from exc

            try:
                strike, direction = _parse_strike_direction(d["question"])
            except ValueError as exc:
                raise ValueError(f"Line {lineno} ({d.get('slug', '?')}): {exc}") from exc

            settlement = _settlement_from_outcome_prices(d["outcomePrices"])
            end_date = d["endDate"]
            start_date = d["startDate"]

            pair_yes = _make_pair(strike, direction, end_date, "YES")
            pair_no = _make_pair(strike, direction, end_date, "NO")

            contracts.append(
                ContractMetadata(
                    id=str(d["id"]),
                    question=d["question"],
                    slug=d["slug"],
                    strike=strike,
                    direction=direction,
                    end_date_utc=end_date,
                    start_date_utc=start_date,
                    settlement=settlement,
                    volume_usd=float(d.get("volume", 0.0)),
                    pair_yes=pair_yes,
                    pair_no=pair_no,
                    raw=d,
                )
            )

    contracts.sort(key=lambda c: c.strike)
    return contracts

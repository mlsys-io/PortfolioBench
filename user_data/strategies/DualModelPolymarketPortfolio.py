"""Dual-model Polymarket portfolio strategy for PortfolioBench.

Behaviour
---------
* Timeframe: 1 hour.
* Decision rule: at each hourly candle, the model computes a fair value for
  each active contract.  A YES position is entered when
  ``ml_edge > MIN_EDGE`` and no position is already open for that pair.
* Sizing: fractional Kelly (``KELLY_FRACTION = 0.15``) applied to free capital.
* Hold rule: once entered, a position is held until the contract expires.
  There is no stop loss, no ROI exit, and no mid-trade rebalancing.
* Settlement: at the expiry candle the trade is force-closed via
  ``custom_exit()`` at the known binary settlement price (0.999 for YES, 0.001
  for NO) via ``custom_exit_price()``.

Data dependencies
-----------------
* Synthetic contract feather files in ``user_data/data/polymarket_ml/``,
  produced by ``polymarket.data_builder.build_all_feathers()``.
* Per-contract event probability CSVs in ``user_data/data/polymarket_ml/``,
  one file per contract named ``{pair}-event_probs.csv``,
  produced by ``scripts/prepare_event_model.py`` (calls
  ``polymarket.data_builder.build_event_predictions()``).
* Contract metadata JSONL in
  ``user_data/data/polymarket_contracts/jan20.jsonl``.

All paths are resolved relative to the freqtrade ``user_data`` directory
stored in ``self.config["user_data_path"]`` when available, falling back to
a path relative to the project root.

Settlement price convention
----------------------------
Freqtrade's ``PRICE_FLOOR`` for Polymarket is 0.001 and ``PRICE_CEIL`` is
0.999.  Returning exactly 0.0 or 1.0 may be clamped by the exchange layer.
We return 0.999 / 0.001 to stay within valid bounds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

from alpha.EventProbAlpha import EventProbAlpha
from polymarket.contracts import ContractMetadata, load_contracts

logger = logging.getLogger(__name__)

# Settlement price constants — must stay within Polymarket exchange bounds
SETTLE_YES = 0.999
SETTLE_NO = 0.001

UTC = timezone.utc


class DualModelPolymarketPortfolio(IStrategy):
    """Buy-and-hold Polymarket YES contracts sized by fractional Kelly.

    See module docstring for full description.
    """

    INTERFACE_VERSION = 3

    timeframe = "1h"
    startup_candle_count: int = 0  # Predictions are precomputed; no warmup needed.

    # Hold until expiry — suppress all automatic exit mechanisms.
    stoploss = -1.0
    minimal_roi = {}
    trailing_stop = False
    use_exit_signal = True     # We need custom_exit() to fire.
    exit_profit_only = False

    # No rebalancing: buy once, hold to expiry.
    position_adjustment_enable = False
    can_short = False

    # Strategy parameters
    MIN_EDGE: float = 0.04          # Minimum model edge to open a position.
    KELLY_FRACTION: float = 0.15    # Fractional Kelly multiplier.
    MAX_ALLOC: float = 0.08         # Cap on any single contract allocation.

    # Per-expiry exposure cap: max fraction of total portfolio on one settlement date.
    MAX_EXPIRY_ALLOC: float = 0.10

    # ---------------------------------------------------------------------------
    # Initialisation helpers
    # ---------------------------------------------------------------------------

    def _resolve_data_root(self) -> Path:
        """Return the absolute path to ``user_data/``."""
        if hasattr(self, "config") and "user_data_path" in self.config:
            return Path(self.config["user_data_path"])
        # Fallback: resolve relative to this file's location (…/user_data/strategies/)
        return Path(__file__).resolve().parents[1]

    def _load_contracts_registry(self) -> dict[str, ContractMetadata]:
        """Load contract metadata and return a dict keyed by pair_yes.

        The JSONL path can be overridden via ``config["contracts_jsonl"]``.
        Defaults to ``user_data/data/polymarket_contracts/jan20.jsonl``.
        """
        data_root = self._resolve_data_root()
        if hasattr(self, "config") and "contracts_jsonl" in self.config:
            jsonl_path = Path(self.config["contracts_jsonl"])
            if not jsonl_path.is_absolute():
                jsonl_path = data_root / jsonl_path
        else:
            jsonl_path = data_root / "data" / "polymarket_contracts" / "jan20.jsonl"
        contracts = load_contracts(jsonl_path, skip_unparseable=True)
        return {c.pair_yes: c for c in contracts}

    def _get_registry(self) -> dict[str, ContractMetadata]:
        if not hasattr(self, "_contract_registry"):
            self._contract_registry = self._load_contracts_registry()
        return self._contract_registry

    def _get_predictions_dir(self) -> Path:
        """Return the directory containing event_probs CSVs.

        Defaults to ``user_data/data/polymarket_ml``.
        Override via ``config["predictions_dir"]`` (relative paths are resolved
        against ``user_data/``).
        """
        data_root = self._resolve_data_root()
        if hasattr(self, "config") and "predictions_dir" in self.config:
            p = Path(self.config["predictions_dir"])
            return p if p.is_absolute() else data_root / p
        return data_root / "data" / "polymarket_ml"

    def _load_event_probs(self, pair: str) -> pd.DataFrame | None:
        """Load per-contract event probability CSV produced by build_event_predictions.

        Returns a DataFrame indexed by UTC timestamps with a ``fair_value`` column,
        or ``None`` if the file does not exist.
        """
        filename = pair.replace("/", "_") + "-event_probs.csv"
        csv_path = self._get_predictions_dir() / filename

        if not csv_path.exists():
            return None

        df = pd.read_csv(str(csv_path), parse_dates=["dt_utc"])
        df = df.set_index("dt_utc")
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def _get_event_probs(self, pair: str) -> pd.DataFrame | None:
        """Lazy-load per-contract event probabilities (cached per pair)."""
        if not hasattr(self, "_event_probs_cache"):
            self._event_probs_cache: dict[str, pd.DataFrame | None] = {}
        if pair not in self._event_probs_cache:
            self._event_probs_cache[pair] = self._load_event_probs(pair)
        return self._event_probs_cache[pair]

    # ---------------------------------------------------------------------------
    # IStrategy interface
    # ---------------------------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata["pair"]
        registry = self._get_registry()

        if pair not in registry:
            logger.warning("Pair %s not found in contract registry — skipping ML alpha.", pair)
            dataframe["ml_fair_value"] = float("nan")
            dataframe["ml_edge"] = 0.0
            dataframe["ml_kelly_alloc"] = 0.0
            dataframe["ml_h_remaining"] = 0.0
            return dataframe

        contract = registry[pair]

        # --- Direct event-probability model ---
        event_probs = self._get_event_probs(pair)
        if event_probs is None:
            logger.warning(
                "No event_probs CSV found for %s — run scripts/prepare_event_model.py first.",
                pair,
            )
            dataframe["ml_fair_value"] = float("nan")
            dataframe["ml_edge"] = 0.0
            dataframe["ml_kelly_alloc"] = 0.0
            dataframe["ml_h_remaining"] = 0.0
            return dataframe

        alpha_meta = {
            "pair": pair,
            "expiry_utc": contract.end_date_utc,
            "event_probs_df": event_probs,
            "kelly_fraction": self.KELLY_FRACTION,
            "min_edge": self.MIN_EDGE,
            "max_alloc": self.MAX_ALLOC,
        }
        return EventProbAlpha(dataframe, alpha_meta).process()

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Enter when there is a positive Kelly allocation above MIN_EDGE.
        entry_condition = (
            (dataframe["ml_kelly_alloc"] > 0.001)
            & (dataframe["ml_edge"] >= self.MIN_EDGE)
            & (dataframe["ml_h_remaining"] > 1.0)    # Don't enter on the expiry candle.
            & (dataframe["close"] > 0.01)
            & (dataframe["close"] < 0.99)
        )
        dataframe.loc[entry_condition, "enter_long"] = 1

        # Encode model probability into enter_tag for reporting.
        dataframe["enter_tag"] = dataframe.apply(
            lambda row: json.dumps(
                {
                    "model_prob": round(float(row.get("ml_fair_value", 0.0)), 4),
                    "edge": round(float(row.get("ml_edge", 0.0)), 4),
                    "kelly": round(float(row.get("ml_kelly_alloc", 0.0)), 4),
                }
            )
            if row.get("enter_long") == 1
            else "",
            axis=1,
        )
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Expiry-based exit is handled in custom_exit().
        # Do not set any exit signals here.
        dataframe["exit_long"] = 0
        return dataframe

    # ---------------------------------------------------------------------------
    # Custom exit: settlement at expiry
    # ---------------------------------------------------------------------------

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        """Trigger settlement exit when the current candle reaches contract expiry."""
        registry = self._get_registry()
        if pair not in registry:
            return None

        contract = registry[pair]
        expiry_ts = pd.Timestamp(contract.end_date_utc, tz="UTC")

        # current_time from freqtrade backtesting is the candle open timestamp.
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=UTC)
        if ct >= expiry_ts:
            return "settlement"

        return None

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ) -> float:
        """Return the binary settlement price when exiting at contract expiry."""
        if exit_tag != "settlement":
            return proposed_rate

        registry = self._get_registry()
        if pair not in registry:
            return proposed_rate

        contract = registry[pair]
        return SETTLE_YES if contract.settlement == 1.0 else SETTLE_NO

    # ---------------------------------------------------------------------------
    # Position sizing: fractional Kelly
    # ---------------------------------------------------------------------------

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        if not self.dp:
            return proposed_stake

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        last = dataframe.iloc[-1]
        kelly_alloc = float(last.get("ml_kelly_alloc", 0.0))
        if kelly_alloc <= 0:
            return 0.0

        # Free capital available for this trade.
        if self.wallets:
            free_capital = float(self.wallets.get_free(self.config["stake_currency"]))
        else:
            free_capital = float(self.config.get("dry_run_wallet", 10000))

        stake = kelly_alloc * free_capital

        # Hard per-expiry cap: total notional on one settlement date must not
        # exceed MAX_EXPIRY_ALLOC × total_portfolio.  We approximate total
        # portfolio as free_capital (this is conservative — open positions
        # are not double-counted).
        registry = self._get_registry()
        if pair in registry:
            target_expiry = registry[pair].end_date_utc
            open_trades = Trade.get_trades_proxy(is_open=True)
            expiry_notional = sum(
                t.open_rate * t.amount
                for t in open_trades
                if t.pair in registry
                and registry[t.pair].end_date_utc == target_expiry
            )
            expiry_cap = self.MAX_EXPIRY_ALLOC * free_capital
            remaining_cap = max(0.0, expiry_cap - expiry_notional)
            stake = min(stake, remaining_cap)

        # Respect freqtrade's min/max stake bounds.
        if min_stake is not None and stake < min_stake:
            return 0.0  # Skip rather than exceed the cap by rounding up.
        return min(stake, max_stake)

    # ---------------------------------------------------------------------------
    # Confirm entry: reject if no ML edge
    # ---------------------------------------------------------------------------

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        registry = self._get_registry()
        if pair not in registry:
            logger.warning("confirm_trade_entry: %s not in registry — rejecting.", pair)
            return False

        contract = registry[pair]
        expiry_ts = pd.Timestamp(contract.end_date_utc, tz="UTC")
        ct = current_time if current_time.tzinfo else current_time.replace(tzinfo=UTC)
        if ct >= expiry_ts:
            logger.info("confirm_trade_entry: %s already expired — rejecting.", pair)
            return False

        # One position per expiry: reject if we already hold any contract
        # with the same settlement date.  Prevents stacking correlated bets
        # (e.g. 88K YES + 90K YES + 92K YES on the same BTC expiry).
        # Trade.get_trades_proxy works in both live and backtesting mode.
        open_trades = Trade.get_trades_proxy(is_open=True)
        for trade in open_trades:
            if trade.pair == pair:
                continue
            if trade.pair in registry:
                other_expiry = registry[trade.pair].end_date_utc
                if other_expiry == contract.end_date_utc:
                    logger.info(
                        "confirm_trade_entry: already holding %s with same expiry %s "
                        "— rejecting %s to prevent correlated-bet stacking.",
                        trade.pair, contract.end_date_utc, pair,
                    )
                    return False

        return True

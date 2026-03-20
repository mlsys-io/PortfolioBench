#!/usr/bin/env python3
"""Generate synthetic Polymarket event contract data for backtesting.

Creates realistic binary outcome contract price data where:
- YES contract prices stay in [0.01, 0.99]
- NO contract = 1 - YES contract (complementary)
- Price paths simulate various event dynamics:
  * Gradual drift (polls slowly shifting)
  * Sharp moves (breaking news)
  * Mean-reversion around equilibrium
  * Resolution convergence (price -> 0 or 1 near expiry)

Usage:
    python utils/generate_polymarket_test_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Event definitions for synthetic data
# ---------------------------------------------------------------------------
EVENTS = [
    {
        "slug": "TRUMP-WIN",
        "description": "Will Trump win the 2024 election?",
        "initial_prob": 0.52,
        "volatility": 0.015,
        "outcome": 1,  # YES wins (resolves to $1)
        "resolution_date": "2025-01-20",
    },
    {
        "slug": "ETH-10K",
        "description": "Will ETH hit $10,000 by end of 2025?",
        "initial_prob": 0.15,
        "volatility": 0.020,
        "outcome": 0,  # NO wins
        "resolution_date": "2025-12-31",
    },
    {
        "slug": "FED-RATE-CUT",
        "description": "Will the Fed cut rates in March 2025?",
        "initial_prob": 0.65,
        "volatility": 0.025,
        "outcome": 1,
        "resolution_date": "2025-03-19",
    },
    {
        "slug": "BTC-100K",
        "description": "Will BTC exceed $100K in 2025?",
        "initial_prob": 0.45,
        "volatility": 0.018,
        "outcome": 1,
        "resolution_date": "2025-12-31",
    },
    {
        "slug": "RECESSION-2025",
        "description": "Will the US enter recession in 2025?",
        "initial_prob": 0.20,
        "volatility": 0.012,
        "outcome": 0,
        "resolution_date": "2025-12-31",
    },
    {
        "slug": "AI-REGULATION",
        "description": "Will major AI regulation pass in 2025?",
        "initial_prob": 0.35,
        "volatility": 0.010,
        "outcome": 0,
        "resolution_date": "2025-12-31",
    },
    {
        "slug": "SPX-6000",
        "description": "Will S&P 500 hit 6000 in 2025?",
        "initial_prob": 0.55,
        "volatility": 0.014,
        "outcome": 1,
        "resolution_date": "2025-12-31",
    },
    {
        "slug": "SOL-500",
        "description": "Will SOL exceed $500 in 2025?",
        "initial_prob": 0.10,
        "volatility": 0.022,
        "outcome": 0,
        "resolution_date": "2025-12-31",
    },
]


def generate_event_prices(
    event: dict,
    start_date: str = "2024-06-01",
    end_date: str = "2026-02-01",
    timeframe: str = "1d",
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic YES and NO contract price data for an event.

    Data extends past the resolution date to *end_date* so that backtests
    with post-resolution timeranges still find data.  After resolution the
    contract trades at its settled value ($0 or $1) with residual volume.

    Returns (yes_df, no_df) each with columns: date, open, high, low, close, volume
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    tf_periods = {
        "5m": 5 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }
    period_ms = tf_periods[timeframe]

    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    resolution_ms = int(datetime.strptime(event["resolution_date"], "%Y-%m-%d").timestamp() * 1000)
    timestamps = list(range(start_ms, end_ms, period_ms))
    n = len(timestamps)

    if n < 10:
        return pd.DataFrame(), pd.DataFrame()

    # Find the index where resolution happens
    resolution_idx = n  # default: resolution is past all data
    for i, ts in enumerate(timestamps):
        if ts >= resolution_ms:
            resolution_idx = i
            break

    # Generate probability path using bounded random walk with drift toward outcome
    prob = event["initial_prob"]
    vol = event["volatility"]
    outcome = event["outcome"]
    probs = np.zeros(n)

    pre_resolution_n = max(resolution_idx, 1)

    for i in range(pre_resolution_n):
        probs[i] = prob

        # Progress toward resolution (0 to 1)
        progress = i / pre_resolution_n

        # Drift toward final outcome increases as resolution approaches
        drift_strength = 0.0001 + 0.005 * (progress ** 3)
        drift = drift_strength * (outcome - prob)

        # Random shocks (occasionally large "news" events)
        shock = rng.normal(0, vol)
        if rng.random() < 0.02:  # 2% chance of "breaking news"
            shock *= 3.0

        # Mean-reversion to prevent extremes before resolution
        if progress < 0.8:
            reversion = 0.001 * (event["initial_prob"] - prob)
        else:
            reversion = 0.0

        prob = prob + drift + shock + reversion
        prob = np.clip(prob, 0.01, 0.99)

    # Force convergence in the last 5% of pre-resolution candles
    convergence_start = int(pre_resolution_n * 0.95)
    for i in range(convergence_start, pre_resolution_n):
        t = (i - convergence_start) / max(pre_resolution_n - convergence_start, 1)
        probs[i] = probs[convergence_start] * (1 - t) + outcome * t
        probs[i] = np.clip(probs[i], 0.01, 0.99)

    # Post-resolution: contract trades at settled value with small noise
    settled_yes = 0.99 if outcome == 1 else 0.01
    for i in range(resolution_idx, n):
        probs[i] = settled_yes + rng.normal(0, 0.002)
        probs[i] = np.clip(probs[i], 0.01, 0.99)

    # Build OHLCV from the probability path
    def build_ohlcv(close_prices: np.ndarray, timestamps: list, rng) -> pd.DataFrame:
        n = len(close_prices)
        spread = np.abs(rng.normal(0, 0.003, n))
        open_prices = close_prices * (1 + rng.normal(0, 0.002, n))
        open_prices = np.clip(open_prices, 0.01, 0.99)
        high_prices = np.minimum(np.maximum(open_prices, close_prices) * (1 + spread), 0.99)
        low_prices = np.maximum(np.minimum(open_prices, close_prices) * (1 - spread), 0.01)
        volume = rng.uniform(1000, 500000, n)

        # Volume tends to increase near resolution
        for i in range(n):
            progress = i / n
            volume[i] *= 1 + 2 * progress

        return pd.DataFrame(
            {
                "date": timestamps,
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            }
        )

    yes_prices = probs
    no_prices = 1.0 - probs  # Complementary contract

    yes_df = build_ohlcv(yes_prices, timestamps, rng)
    no_df = build_ohlcv(no_prices, timestamps, rng)

    return yes_df, no_df


def main():
    # Output to both polymarket/ (for polymarket config) and portfoliobench/ (for default config)
    output_dirs = [
        Path("user_data/data/polymarket"),
        Path("user_data/data/portfoliobench"),
    ]
    for d in output_dirs:
        d.mkdir(parents=True, exist_ok=True)

    timeframes = ["5m", "4h", "1d"]
    count = 0

    np.random.seed(42)

    for event in EVENTS:
        for tf in timeframes:
            seed = hash(f"{event['slug']}-{tf}") % (2**31)
            yes_df, no_df = generate_event_prices(event, timeframe=tf, seed=seed)

            if yes_df.empty:
                print(f"[SKIP] {event['slug']} {tf}: insufficient data range")
                continue

            slug = event["slug"]

            for data_dir in output_dirs:
                # Save YES contract
                yes_path = data_dir / f"{slug}-YES_USDT-{tf}.feather"
                yes_df.to_feather(yes_path, compression_level=9, compression="lz4")

                # Save NO contract
                no_path = data_dir / f"{slug}-NO_USDT-{tf}.feather"
                no_df.to_feather(no_path, compression_level=9, compression="lz4")

            count += 2

            print(
                f"[DATA] {slug} {tf}: YES={len(yes_df)} rows "
                f"(p0={yes_df['close'].iloc[0]:.3f} -> p_end={yes_df['close'].iloc[-1]:.3f}), "
                f"NO={len(no_df)} rows"
            )

    print(f"\nGenerated {count} feather files in {', '.join(str(d) for d in output_dirs)}")


if __name__ == "__main__":
    main()

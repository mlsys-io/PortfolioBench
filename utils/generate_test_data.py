#!/usr/bin/env python3
"""Generate synthetic OHLCV feather data files for backtesting.

Run this script when the LFS-tracked feather files are not available
(e.g., pointer stubs only) to generate realistic synthetic data that
allows all backtest tests to pass.

Usage:
    python utils/generate_test_data.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def main():
    data_dir = Path("user_data/data/binance")
    files = list(data_dir.glob("*.feather"))

    # Known approximate prices for key tickers
    ticker_prices = {
        'BTC': 95000, 'ETH': 3400, 'SOL': 190, 'XRP': 2.2, 'ADA': 0.9,
        'AAPL': 230, 'MSFT': 420, 'NVDA': 140, 'GOOG': 190, 'AMZN': 220,
        'DJI': 42000, 'FTSE': 8400, 'GSPC': 5900,
        'AMD': 120, 'AVGO': 180, 'META': 550, 'TSLA': 350, 'JPM': 230,
    }

    tf_periods = {
        "5m": 5 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }

    start_ms = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ms = int(datetime(2026, 2, 1).timestamp() * 1000)

    np.random.seed(42)
    count = 0

    for fpath in sorted(files):
        fname = fpath.name
        parts = fname.replace('.feather', '').rsplit('-', 1)
        if len(parts) != 2:
            continue
        pair_str, tf = parts
        if tf not in tf_periods:
            continue

        ticker = pair_str.replace('_USDT', '').replace('_USD', '')
        period_ms = tf_periods[tf]

        timestamps = list(range(start_ms, end_ms, period_ms))
        n = len(timestamps)

        # Use known price or random between 50-500
        base_price = ticker_prices.get(ticker, np.random.uniform(50, 500))

        # Mean-reverting random walk
        returns = np.random.normal(0, 0.002, n)
        prices = np.zeros(n)
        prices[0] = base_price
        for i in range(1, n):
            reversion = 0.0001 * (base_price - prices[i - 1]) / base_price
            prices[i] = prices[i - 1] * (1 + returns[i] + reversion)

        close_prices = prices
        spread = np.abs(np.random.normal(0, 0.001, n))
        open_prices = close_prices * (1 + np.random.normal(0, 0.001, n))
        high_prices = np.maximum(open_prices, close_prices) * (1 + spread)
        low_prices = np.minimum(open_prices, close_prices) * (1 - spread)
        volume = np.random.uniform(100, 1000000, n)

        df = pd.DataFrame({
            'date': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
        })

        df.to_feather(fpath, compression_level=9, compression="lz4")
        count += 1

    print(f"Generated {count} feather files in {data_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic OHLCV feather data files for backtesting.

Run this script when the Google Drive feather files have not been downloaded
to generate realistic synthetic data that allows all backtest tests to pass.

To download real data instead, run:
    portbench download-data --exchange portfoliobench

Generates data in both user_data/data/usstock/ (for direct data loading)
and user_data/data/portfoliobench/ (for freqtrade backtesting with the
portfoliobench exchange).

Usage:
    portbench generate-data                          # via CLI (recommended)
    python utils/generate_test_data.py               # or directly
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# All tickers tracked in the repo (119 instruments)
CRYPTO_TICKERS = [
    "ADA_USDT", "BCH_USDT", "BNB_USDT", "BTC_USDT", "DOGE_USDT",
    "ETH_USDT", "SOL_USDT", "STETH_USDT", "TRX_USDT", "XRP_USDT",
]

STOCK_INDEX_TICKERS = [
    "AAPL_USD", "ABBV_USD", "ABT_USD", "ADBE_USD", "ADI_USD", "AMAT_USD",
    "AMD_USD", "AMGN_USD", "AMZN_USD", "ANET_USD", "APH_USD", "APP_USD",
    "AVGO_USD", "AXP_USD", "BAC_USD", "BA_USD", "BKNG_USD", "BLK_USD",
    "BMY_USD", "BRK-B_USD", "BX_USD", "CAT_USD", "CMCSA_USD", "COF_USD",
    "COP_USD", "COST_USD", "CRM_USD", "CSCO_USD", "CVX_USD", "C_USD",
    "DE_USD", "DHR_USD", "DIS_USD", "DJI_USD", "FTSE_USD", "GEV_USD",
    "GE_USD", "GILD_USD", "GOOG_USD", "GSPC_USD", "GS_USD", "HCA_USD",
    "HD_USD", "HON_USD", "HSI_USD", "IBKR_USD", "IBM_USD", "INTC_USD",
    "INTU_USD", "ISRG_USD", "IXIC_USD", "JNJ_USD", "JPM_USD", "KLAC_USD",
    "KO_USD", "LLY_USD", "LMT_USD", "LOW_USD", "LRCX_USD", "MA_USD",
    "MCD_USD", "MCK_USD", "META_USD", "MRK_USD", "MSFT_USD", "MS_USD",
    "MU_USD", "N225_USD", "NEE_USD", "NEM_USD", "NFLX_USD", "NOW_USD",
    "NVDA_USD", "ORCL_USD", "PANW_USD", "PEP_USD", "PFE_USD", "PGR_USD",
    "PG_USD", "PH_USD", "PLD_USD", "PLTR_USD", "PM_USD", "QCOM_USD",
    "RTX_USD", "RUT_USD", "SBUX_USD", "SCCO_USD", "SCHW_USD", "SPGI_USD",
    "STOXX50E_USD", "SYK_USD", "TJX_USD", "TMO_USD", "TMUS_USD",
    "TSLA_USD", "TXN_USD", "T_USD", "UBER_USD", "UNH_USD", "UNP_USD",
    "VIX_USD", "VRTX_USD", "VZ_USD", "V_USD", "WELL_USD", "WFC_USD",
    "WMT_USD", "XOM_USD",
]

POLYMARKET_TICKERS = [
    "TRUMP-WIN-YES_USDT", "TRUMP-WIN-NO_USDT",
    "ETH-10K-YES_USDT", "ETH-10K-NO_USDT",
    "FED-RATE-CUT-YES_USDT", "FED-RATE-CUT-NO_USDT",
    "BTC-100K-YES_USDT", "BTC-100K-NO_USDT",
    "RECESSION-2025-YES_USDT", "RECESSION-2025-NO_USDT",
    "AI-REGULATION-YES_USDT", "AI-REGULATION-NO_USDT",
    "SPX-6000-YES_USDT", "SPX-6000-NO_USDT",
    "SOL-500-YES_USDT", "SOL-500-NO_USDT",
]

ALL_TICKERS = CRYPTO_TICKERS + STOCK_INDEX_TICKERS
ALL_TICKERS_WITH_POLYMARKET = ALL_TICKERS + POLYMARKET_TICKERS

TIMEFRAMES = ["5m", "4h", "1d"]

# Known approximate prices for key tickers
TICKER_PRICES = {
    'BTC': 95000, 'ETH': 3400, 'SOL': 190, 'XRP': 2.2, 'ADA': 0.9,
    'AAPL': 230, 'MSFT': 420, 'NVDA': 140, 'GOOG': 190, 'AMZN': 220,
    'DJI': 42000, 'FTSE': 8400, 'GSPC': 5900,
    'AMD': 120, 'AVGO': 180, 'META': 550, 'TSLA': 350, 'JPM': 230,
}

# Polymarket contract prices (probabilities between 0 and 1)
POLYMARKET_PRICES = {
    'TRUMP-WIN-YES': 0.55, 'TRUMP-WIN-NO': 0.45,
    'ETH-10K-YES': 0.20, 'ETH-10K-NO': 0.80,
    'FED-RATE-CUT-YES': 0.65, 'FED-RATE-CUT-NO': 0.35,
    'BTC-100K-YES': 0.70, 'BTC-100K-NO': 0.30,
    'RECESSION-2025-YES': 0.25, 'RECESSION-2025-NO': 0.75,
    'AI-REGULATION-YES': 0.40, 'AI-REGULATION-NO': 0.60,
    'SPX-6000-YES': 0.45, 'SPX-6000-NO': 0.55,
    'SOL-500-YES': 0.15, 'SOL-500-NO': 0.85,
}


def _timeframe_periods_ms():
    return {
        "5m": 5 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
    }


def generate_ohlcv(ticker, tf, rng):
    """Generate a synthetic OHLCV DataFrame for one ticker/timeframe."""
    tf_periods_ms = _timeframe_periods_ms()

    start_ms = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ms = int(datetime(2026, 2, 1).timestamp() * 1000)
    period_ms = tf_periods_ms[tf]

    timestamps = list(range(start_ms, end_ms, period_ms))
    n = len(timestamps)

    base_price = TICKER_PRICES.get(ticker, rng.uniform(50, 500))

    # Mean-reverting random walk
    returns = rng.normal(0, 0.002, n)
    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        reversion = 0.0001 * (base_price - prices[i - 1]) / base_price
        prices[i] = prices[i - 1] * (1 + returns[i] + reversion)

    close_prices = prices
    spread = np.abs(rng.normal(0, 0.001, n))
    open_prices = close_prices * (1 + rng.normal(0, 0.001, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + spread)
    low_prices = np.minimum(open_prices, close_prices) * (1 - spread)
    volume = rng.uniform(100, 1000000, n)

    return pd.DataFrame({
        'date': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
    })


def generate_polymarket_ohlcv(ticker, tf, rng):
    """Generate synthetic OHLCV data for a Polymarket prediction contract.

    Prices are bounded between 0.001 and 0.999 (probability space).
    """
    tf_periods_ms = _timeframe_periods_ms()

    start_ms = int(datetime(2024, 1, 1).timestamp() * 1000)
    end_ms = int(datetime(2026, 2, 1).timestamp() * 1000)
    period_ms = tf_periods_ms[tf]

    timestamps = list(range(start_ms, end_ms, period_ms))
    n = len(timestamps)

    base_price = POLYMARKET_PRICES.get(ticker, rng.uniform(0.2, 0.8))

    # Mean-reverting random walk bounded to [0.001, 0.999]
    returns = rng.normal(0, 0.001, n)
    prices = np.zeros(n)
    prices[0] = base_price
    for i in range(1, n):
        reversion = 0.001 * (base_price - prices[i - 1])
        prices[i] = prices[i - 1] + returns[i] + reversion
        prices[i] = np.clip(prices[i], 0.001, 0.999)

    close_prices = prices
    spread = np.abs(rng.normal(0, 0.0005, n))
    open_prices = np.clip(close_prices + rng.normal(0, 0.0005, n), 0.001, 0.999)
    high_prices = np.clip(np.maximum(open_prices, close_prices) + spread, 0.001, 0.999)
    low_prices = np.clip(np.minimum(open_prices, close_prices) - spread, 0.001, 0.999)
    volume = rng.uniform(1000, 500000, n)

    return pd.DataFrame({
        'date': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume,
    })


def main():
    # Output directories: usstock (for direct loading) + portfoliobench (for freqtrade backtests)
    output_dirs = [
        Path("user_data/data/usstock"),
        Path("user_data/data/portfoliobench"),
    ]
    # Polymarket data goes into portfoliobench (for default config) and polymarket (for polymarket config)
    polymarket_dirs = [
        Path("user_data/data/portfoliobench"),
        Path("user_data/data/polymarket"),
    ]
    for d in set(output_dirs + polymarket_dirs):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    count = 0

    for pair_str in ALL_TICKERS:
        ticker = pair_str.replace('_USDT', '').replace('_USD', '')
        for tf in TIMEFRAMES:
            fname = f"{pair_str}-{tf}.feather"
            df = generate_ohlcv(ticker, tf, rng)

            for data_dir in output_dirs:
                fpath = data_dir / fname
                df.to_feather(fpath, compression_level=9, compression="lz4")

            count += 1

    # Generate Polymarket prediction market data
    for pair_str in POLYMARKET_TICKERS:
        ticker = pair_str.replace('_USDT', '')
        for tf in TIMEFRAMES:
            fname = f"{pair_str}-{tf}.feather"
            df = generate_polymarket_ohlcv(ticker, tf, rng)

            for data_dir in polymarket_dirs:
                fpath = data_dir / fname
                df.to_feather(fpath, compression_level=9, compression="lz4")

            count += 1

    print(f"Generated {count} feather files in {', '.join(str(d) for d in set(output_dirs + polymarket_dirs))}")


if __name__ == "__main__":
    main()

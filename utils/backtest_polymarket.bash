#!/bin/bash

# ============================================================================
# Polymarket Event Contract Backtesting
# ============================================================================
# Backtests prediction market strategies on binary outcome contracts.
# Run generate_polymarket_test_data.py first to create synthetic event data.

# 1. Configuration
OUTPUT_DIR="user_data/backtest_results/polymarket"
DATA_DIR="user_data/data/polymarket"
CONFIG="user_data/config_polymarket.json"

mkdir -p "$OUTPUT_DIR"

# 2. Define Event Contract Pairs
# YES contracts for various events
POLITICAL_PAIRS="TRUMP-WIN-YES/USDT FED-RATE-CUT-YES/USDT RECESSION-2025-YES/USDT AI-REGULATION-YES/USDT"
CRYPTO_PAIRS="ETH-10K-YES/USDT BTC-100K-YES/USDT SOL-500-YES/USDT"
MARKET_PAIRS="SPX-6000-YES/USDT"
ALL_YES_PAIRS="$POLITICAL_PAIRS $CRYPTO_PAIRS $MARKET_PAIRS"

# 3. Define Strategies to Test
STRATEGIES=("PolymarketMomentumStrategy" "PolymarketMeanReversionStrategy")
PORTFOLIO_STRATEGY="PolymarketPortfolio"

# 4. Define Timeframes
TIMEFRAMES=("4h" "1d")

# 5. Core Execution Function
run_backtest() {
    local strategy=$1
    local strategy_path=$2
    local category=$3
    local tf=$4
    local pairs=$5

    local timerange=""
    if [[ "$tf" == "5m" ]]; then
        timerange="20240601-20241231"
    elif [[ "$tf" == "4h" ]]; then
        timerange="20240601-20251201"
    elif [[ "$tf" == "1d" ]]; then
        timerange="20240601-20251201"
    fi

    local export_file="${OUTPUT_DIR}/${strategy}_${category}_${tf}.json"

    echo "=================================================="
    echo "Strategy: $strategy | Category: $category | TF: $tf"
    echo "Timerange: $timerange"
    echo "Pairs: $pairs"
    echo "=================================================="

    freqtrade backtesting \
        --config "$CONFIG" \
        --strategy "$strategy" \
        --strategy-path "$strategy_path" \
        --timeframe "$tf" \
        --timerange "$timerange" \
        --datadir "$DATA_DIR" \
        --pairs $pairs \
        --dry-run-wallet 10000 \
        --export trades \
        --export-filename "$export_file"

    echo "-> Saved results to $export_file"
    echo ""
}

# 6. Run trading strategy backtests
for strategy in "${STRATEGIES[@]}"; do
    for tf in "${TIMEFRAMES[@]}"; do
        run_backtest "$strategy" "./strategy" "political" "$tf" "$POLITICAL_PAIRS"
        run_backtest "$strategy" "./strategy" "crypto_events" "$tf" "$CRYPTO_PAIRS"
        run_backtest "$strategy" "./strategy" "all_events" "$tf" "$ALL_YES_PAIRS"
    done
done

# 7. Run portfolio strategy backtests
for tf in "${TIMEFRAMES[@]}"; do
    run_backtest "$PORTFOLIO_STRATEGY" "./user_data/strategies" "portfolio_all" "$tf" "$ALL_YES_PAIRS"
done

echo "All Polymarket backtests completed! Check $OUTPUT_DIR."

#!/bin/bash

# 1. Configuration
STRATEGY="ONS_Portfolio" # Update this to the exact name of your strategy class
OUTPUT_DIR="user_data/backtest_results"

# Create the output directory if it doesn't already exist
mkdir -p "$OUTPUT_DIR"

# 2. Define Asset Categories 
CRYPTO_PAIRS="BTC/USDT ETH/USDT SOL/USDT XRP/USDT"
STOCK_PAIRS="AAPL/USDT MSFT/USDT NVDA/USDT GOOG/USDT"
INDEX_PAIRS="DJI/USDT FTSE/USDT GSPC/USDT"
MIX_PAIRS="$CRYPTO_PAIRS $STOCK_PAIRS $INDEX_PAIRS"

# 3. Define Timeframes
TIMEFRAMES=("5m" "1d")

# 4. Core Execution Function
run_backtest() {
    local category=$1
    local tf=$2
    local pairs=$3
    
    # if [[ "$category" != "index_only" ]]; then
    #     echo "Skipping $category for $tf as requested..."
    #     echo "=================================================="
    #     echo ""
    #     return
    # fi
    
    # Set the timerange based on the timeframe
    local timerange=""
    if [[ "$tf" == "5m" ]]; then
        timerange="20260101-20260108"
    elif [[ "$tf" == "1d" ]]; then
        timerange="20240101-20260131"
    fi
    
    # Construct a clean output filename
    local export_file="${OUTPUT_DIR}/${category}_${tf}.json"
    
    echo "=================================================="
    echo "Starting backtest | Category: $category | Timeframe: $tf | Timerange: $timerange"
    echo "Pairs: $pairs"
    echo "=================================================="
    
    # Execute Freqtrade via Docker
    docker compose run --rm freqtrade backtesting \
        --strategy "$STRATEGY" \
        --timeframe "$tf" \
        --timerange "$timerange" \
        --pairs $pairs \
        --export trades \
        --export-filename "$export_file"
        
    echo "-> Saved results to $export_file"
    echo ""
}

# 5. Loop through categories and timeframes
for tf in "${TIMEFRAMES[@]}"; do
    run_backtest "crypto_only" "$tf" "$CRYPTO_PAIRS"
    run_backtest "stock_only" "$tf" "$STOCK_PAIRS"
    run_backtest "index_only" "$tf" "$INDEX_PAIRS"
    run_backtest "mix_assets" "$tf" "$MIX_PAIRS"
done

echo "All backtests completed successfully! Check the $OUTPUT_DIR folder."
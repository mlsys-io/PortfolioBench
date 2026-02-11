#!/usr/bin/bash

# A toy script to set up simple logic for backtesting. Will upgrade to a more customized one later

if [[ ! -n "$VIRTUAL_ENV" ]]; then
    echo "Virtual environment is not active. Activating virtual environment"
    source .venv/bin/activate
fi

# Test
freqtrade --version
echo "-s strategy"
echo "-d data directory"
echo "-a assets"

# Backtesting
echo "Start backtesting:"

# Collect user defined flags
optstring="s:d:a:"
pairsxs=""
strategy=""
datapath=""

while getopts "$optstring" flag; do 
    case "$flag" in
        s) strategy+="$OPTARG ";;
        d) datapath=$OPTARG;;
        a) pairxs+="$OPTARG ";;
    esac
done

# Build backtesting command flag

cli_flags="--timerange 20250501-20250601 --timeframe 4h --strategy-path ./strategy "

if [[ "$strategy" !=  "" ]]; then
    cli_flags+="--strategy $strategy "
fi

if [[ "$pairxs" != "" ]]; then
    cli_flags+="--pairs $pairxs"
fi

# Run freqtrade backtesting
echo $cli_flags
freqtrade backtesting $cli_flags
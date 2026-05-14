import json
import numpy as np
import pandas as pd

import json

# Define a helper function to safely parse fields that are JSON-encoded strings.
def parse_json_list(value):
    # If the value is already a list, return it directly.
    if isinstance(value, list):
        return value
    
    # If the value is missing, return an empty list.
    if value is None:
        return []
    
    # If the value is a string, try to parse it as JSON.
    if isinstance(value, str):
        # Remove surrounding whitespace.
        value = value.strip()
        
        # Return an empty list for blank strings.
        if value == "":
            return []
        
        # Parse the string as JSON.
        return json.loads(value)
    
    # For anything unexpected, return an empty list.
    return []


# Define a helper function to convert fee strings into decimal form.
def parse_fee_decimal(value):
    if value is None:
        return np.nan
    try:
        s = str(value).strip()
        if s == "":
            return np.nan

        # If already a decimal string like "0.02"
        if "." in s or "e" in s.lower():
            return float(s)

        # If integer-like and likely wei-scaled
        i = int(s)
        return i / 1e18 if i > 1 else float(i)
    except Exception:
        return np.nan


# Define a helper function to safely convert values to float.
def safe_float(value):
    # Return NaN if the value is missing.
    if value is None:
        return np.nan
    
    # Try to cast the value to float.
    try:
        return float(value)
    except Exception:
        return np.nan


def read_market_to_df(market_path) -> pd.DataFrame:
    # Create an empty list to collect market-level rows.
    market_rows = []

    # Open the market file for reading.
    with open(market_path, "r", encoding="utf-8") as f:
        # Loop through each line in the file.
        for line in f:
            # Parse the current line as JSON.
            market = json.loads(line)
            
            # Parse the outcomes field, which is stored as a JSON string.
            outcomes = parse_json_list(market.get("outcomes"))
            
            # Parse the outcomePrices field, which is stored as a JSON string.
            outcome_prices = parse_json_list(market.get("outcomePrices"))
            
            # Build a clean market-level row.
            market_rows.append({
                "market_id": market.get("id"),
                "condition_id": market.get("conditionId"),
                "question": market.get("question"),
                "slug": market.get("slug"),
                "description": market.get("description"),
                "start_date": market.get("startDate"),
                "end_date": market.get("endDate"),
                "closed_time": market.get("closedTime"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "archived": market.get("archived"),
                "accepting_orders": market.get("acceptingOrders"),
                "enable_order_book": market.get("enableOrderBook"),
                "neg_risk": market.get("negRisk"),
                "fee_decimal": parse_fee_decimal(market.get("fee")),
                "volume": safe_float(market.get("volume")),
                "volume_clob": safe_float(market.get("volumeClob")),
                "market_best_bid": safe_float(market.get("bestBid")),
                "market_best_ask": safe_float(market.get("bestAsk")),
                "last_trade_price": safe_float(market.get("lastTradePrice")),
                "resolution_source": market.get("resolutionSource"),
                "uma_resolution_status": market.get("umaResolutionStatus"),
                "created_at": market.get("createdAt"),
                "updated_at": market.get("updatedAt"),
                "n_outcomes": len(outcomes),
                "outcomes_raw": json.dumps(outcomes),
                "outcome_prices_raw": json.dumps(outcome_prices),
            })

    # Convert the list of dicts into a pandas DataFrame.
    markets_df = pd.DataFrame(market_rows)
    return markets_df

def read_token_to_df(token_path) -> pd.DataFrame:
    market_token_rows = []

    with open(token_path, "r", encoding="utf-8") as f:
        for line in f:
            market = json.loads(line)

            outcomes = parse_json_list(market.get("outcomes"))
            outcome_prices = parse_json_list(market.get("outcomePrices"))
            token_ids = parse_json_list(market.get("clobTokenIds"))

            for i, token_id in enumerate(token_ids):
                market_token_rows.append({
                    "market_id": market.get("id"),
                    "token_id": str(token_id),
                    "outcome_index": i,
                    "outcome": outcomes[i] if i < len(outcomes) else None,
                    "outcome_price_initial": safe_float(outcome_prices[i]) if i < len(outcome_prices) else np.nan,
                    "question": market.get("question"),
                    "slug": market.get("slug"),
                    "description": market.get("description"),
                    "start_date": market.get("startDate"),
                    "end_date": market.get("endDate"),
                    "closed_time": market.get("closedTime"),
                    "active": market.get("active"),
                    "closed": market.get("closed"),
                    "archived": market.get("archived"),
                    "accepting_orders": market.get("acceptingOrders"),
                    "enable_order_book": market.get("enableOrderBook"),
                    "neg_risk": market.get("negRisk"),
                    "fee_decimal": parse_fee_decimal(market.get("fee")),
                    "volume": safe_float(market.get("volume")),
                    "volume_clob": safe_float(market.get("volumeClob")),
                })

    market_tokens_df = pd.DataFrame(market_token_rows)
    return market_tokens_df

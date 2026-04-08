"""Polymarket backtesting subsystem for PortfolioBench.

Modules
-------
contracts        ContractMetadata dataclass and JSONL loader.
settlement       Derive binary settlement from historical BTC data.
synthetic_prices Generate synthetic hourly OHLCV for contracts lacking real data.
event_features   Feature engineering for the direct event-probability model.
event_dataset    Build training samples for the event-probability model.
event_model      Train, calibrate, and run inference with the event-probability model.
data_builder     Orchestrator: contracts → feathers → training data → predictions.
"""

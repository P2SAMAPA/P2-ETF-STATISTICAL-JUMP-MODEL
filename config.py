"""
Configuration for Statistical Jump Model (SJM) engine.
"""

import os
from datetime import datetime

# Hugging Face
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-sjm-results"   # create this repo

# Universe definitions (same as before)
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# SJM hyperparameters – tuned for long, persistent regimes
LAMBDA = 10.0          # penalty for second difference (regime changes)
GAMMA = 20.0           # penalty for first difference (short regimes)
MIN_REGIME_DAYS = 60   # any regime shorter than this will be merged
TRANSITION_THRESHOLD = 0.5   # std dev threshold for changepoint detection
VOL_WINDOW = 30        # rolling volatility window (days)

# Output
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

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

# Which series to model? "returns", "volatility", or "macro"
SERIES_TYPE = "returns"   # we will use ETF log‑returns

# SJM hyperparameters (to be tuned)
LAMBDA = 0.1      # penalty for second difference (regime changes)
GAMMA = 1.0       # penalty for first difference (persistence)
MIN_REGIME_DAYS = 60   # minimum duration to consider a regime valid
TRANSITION_THRESHOLD = 0.5  # how many std devs to call a jump

# Output
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

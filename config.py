"""config.py — Statistical Jump Model configuration."""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-sjm-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWF",
    "XSD",
    "XBI",
    "IWM",
]
COMBINED_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": COMBINED_TICKERS,
}

# ── SJM hyperparameters ───────────────────────────────────────────────────────
LAMBDA = 10.0  # second-difference penalty (regime change cost)
GAMMA = 20.0  # first-difference penalty (persistence)
MIN_REGIME_DAYS = 60  # merge regimes shorter than this
TRANSITION_THRESHOLD = 0.5  # std-dev multiplier for changepoint detection
VOL_WINDOW = 30  # rolling volatility window (days)
N_REGIMES = 3  # number of broad regimes (low / mid / high)

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")

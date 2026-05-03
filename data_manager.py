"""data_manager.py — Data loading and preprocessing for SJM engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config


def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)

    # Normalise index → Date column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows × {len(df.columns)} cols")
    return df


def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Compute log returns from closing prices (data is already prices)."""
    available = [t for t in tickers if t in df_wide.columns]
    prices = df_wide.set_index("Date")[available].copy()
    prices = prices.ffill()  # forward-fill any gaps before computing returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns[available]

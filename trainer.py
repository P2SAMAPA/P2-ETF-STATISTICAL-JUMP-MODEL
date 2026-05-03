"""trainer.py — SJM engine orchestrator."""

from __future__ import annotations

import os

import numpy as np

import config
import data_manager
import push_results
from regime_detector import StatisticalJumpModel


def process_ticker(ticker: str, series: "pd.Series") -> dict | None:  # noqa: F821
    """Fit SJM to one ticker and return structured results."""

    if len(series) < config.MIN_REGIME_DAYS * 2:
        print(f"    {ticker}: insufficient data ({len(series)} rows) — skipping")
        return None

    model = StatisticalJumpModel(
        lambda_pen=config.LAMBDA,
        gamma_persist=config.GAMMA,
        min_regime_days=config.MIN_REGIME_DAYS,
        transition_threshold=config.TRANSITION_THRESHOLD,
        vol_window=config.VOL_WINDOW,
        n_regimes=config.N_REGIMES,
    )
    model.fit(series.values)

    curr_regime = model.get_current_regime()
    curr_return = model.regime_return_map_.get(curr_regime, 0.0)

    # Regime return summary for printing
    for reg_id, ann_ret in sorted(model.regime_return_map_.items()):
        label = model.get_regime_name(reg_id)
        print(
            f"    {ticker}  Regime {reg_id} ({label:3s}): "
            f"annualised median return = {ann_ret*100:+.2f}%"
            f"{'  ← CURRENT' if reg_id == curr_regime else ''}"
        )

    transitions = [
        {"date": series.index[idx].strftime("%Y-%m-%d"), "new_regime": reg}
        for idx, reg in model.get_transitions()
    ]

    return {
        "current_regime": curr_regime,
        "current_regime_name": model.get_regime_name(curr_regime),
        "current_duration_days": model.get_current_duration(),
        "current_annual_return": round(curr_return, 6),
        "total_segments": len(model.durations_),
        "average_duration_days": int(np.mean(model.durations_)),
        "regime_returns": {
            int(k): round(v, 6) for k, v in model.regime_return_map_.items()
        },
        "regime_names": {
            int(k): model.get_regime_name(k) for k in model.regime_return_map_
        },
        "regime_sequence": [int(x) for x in model.regime_labels_],
        "dates": [d.strftime("%Y-%m-%d") for d in series.index],
        "transition_points": transitions,
        "n_changepoints": len(model.changepoints_),
    }


def main() -> None:
    if not config.HF_TOKEN:
        print("HF_TOKEN not set — aborting.")
        return

    # Which universe to run (set by GitHub Actions matrix)
    target = os.environ.get("SJM_UNIVERSE", "ALL").upper()

    df = data_manager.load_master_data()

    all_results: dict = {}

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue

        print(f"\n{'='*60}")
        print(f"Universe: {universe_name}  ({len(tickers)} tickers)")
        print(f"{'='*60}")

        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            print("  No data — skipping.")
            continue

        universe_results: dict = {}
        rankings: list[tuple[str, float]] = []

        for ticker in returns.columns:
            print(f"\n  {ticker}")
            series = returns[ticker].dropna()
            result = process_ticker(ticker, series)
            if result is None:
                continue
            universe_results[ticker] = result
            rankings.append((ticker, result["current_annual_return"]))

        # Rank tickers by current regime's annualised return
        rankings.sort(key=lambda x: x[1], reverse=True)
        for rank, (ticker, ann_ret) in enumerate(rankings, 1):
            universe_results[ticker]["rank"] = rank
            print(
                f"  Rank {rank:2d}  {ticker:6s}  "
                f"Regime={universe_results[ticker]['current_regime_name']:3s}  "
                f"AnnReturn={ann_ret*100:+.2f}%"
            )

        all_results[universe_name] = {
            "rankings": rankings,
            "tickers": universe_results,
        }

    output = {"run_date": config.TODAY, "universes": all_results}
    push_results.push_daily_result(output, universe=target)
    print("\n✅ SJM analysis complete — results pushed to HuggingFace.")


if __name__ == "__main__":
    main()

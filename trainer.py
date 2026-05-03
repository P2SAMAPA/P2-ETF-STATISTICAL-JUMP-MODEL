"""
Main trainer: for each universe, for each ticker, fit SJM and collect results.
Also computes average return per regime for return chasing.
"""

import pandas as pd
import numpy as np
import config
import data_manager
from regime_detector import StatisticalJumpModel
import push_results

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty:
            continue

        universe_results = {}
        for ticker in returns.columns:
            print(f"  Processing {ticker}...")
            series = returns[ticker].dropna()
            if len(series) < config.MIN_REGIME_DAYS:
                continue

            model = StatisticalJumpModel(
                lambda_pen=config.LAMBDA,
                gamma_persist=config.GAMMA,
                min_regime_days=config.MIN_REGIME_DAYS,
                transition_threshold=config.TRANSITION_THRESHOLD,
                vol_window=config.VOL_WINDOW
            )
            model.fit(series.values)

            # Compute average return per regime (annualised, simple)
            regime_returns = {}
            for regime_id in set(model.regime_labels_):
                mask = model.regime_labels_ == regime_id
                avg_daily = series.values[mask].mean()
                # Annualise (252 trading days)
                annual_return = (1 + avg_daily) ** 252 - 1
                regime_returns[int(regime_id)] = annual_return

            # Build output
            transitions_dates = []
            for idx, new_reg in model.get_transitions():
                date = series.index[idx].strftime("%Y-%m-%d")
                transitions_dates.append({"date": date, "new_regime": int(new_reg)})

            universe_results[ticker] = {
                "current_regime": int(model.get_current_regime()),
                "current_duration_days": int(model.get_current_duration()),
                "total_regimes": len(model.durations_),
                "average_duration_days": int(np.mean(model.durations_)),
                "regime_sequence": [int(l) for l in model.regime_labels_],
                "dates": [d.strftime("%Y-%m-%d") for d in series.index],
                "transition_points": transitions_dates,
                "regime_returns": regime_returns   # new field
            }

        all_results[universe_name] = universe_results

    output = {"run_date": config.TODAY, "universes": all_results}
    push_results.push_daily_result(output)
    print("\n=== SJM analysis complete, results pushed ===")

if __name__ == "__main__":
    main()

# P2-ETF-SJM — Statistical Jump Model for Regime Detection

Detects persistent regime switches (bull/bear/crisis) in ETF returns using convex relaxation.  
Outputs: current regime, duration, transition points.

## Methodology

We model daily log‑returns as a piecewise constant Gaussian process with two penalties:
- **Regime change penalty** (second difference) – encourages flat segments.
- **Persistence penalty** (first difference) – discourages very short regimes.

The problem is solved via **quadratic programming (cvxpy)**.

### References

- Nystrup, P., et al. (2020) – *"Regime-based versus static asset allocation: Letting the data speak"* (original SJM formulation)
- Bemporad, A., et al. (2018) – *"A fitted value iteration method for biclustering"* (QP relaxation)
- Bai, J., & Perron, P. (1998) – *"Estimating and testing linear models with multiple structural changes"*
- Killick, R., et al. (2012) – *"Optimal detection of changepoints with a linear computational cost"* (PELT)

## Files

- `config.py` : parameters (λ, γ, min regime duration)
- `regime_detector.py` : SJM class with cvxpy
- `trainer.py` : run on ETF universes
- `app.py` : Streamlit dashboard (regime status, timeline, transitions)

## Run locally

```bash
pip install -r requirements.txt
python trainer.py
streamlit run app.py
GitHub Actions
Daily run at 6 AM UTC. Results stored in P2SAMAPA/p2-etf-sjm-results.

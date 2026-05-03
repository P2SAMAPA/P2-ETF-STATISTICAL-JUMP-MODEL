"""
Statistical Jump Model via convex relaxation (fused LASSO + persistence).
"""

import numpy as np
from sklearn.cluster import KMeans

class StatisticalJumpModel:
    def __init__(self, lambda_pen=0.1, gamma_persist=1.0, min_regime_days=60, transition_threshold=0.5):
        self.lambda_ = lambda_pen
        self.gamma_ = gamma_persist
        self.min_regime = min_regime_days
        self.threshold = transition_threshold
        self.mu_ = None
        self.log_sigma_ = None
        self.changepoints_ = None
        self.regime_labels_ = None
        self.durations_ = None

    def fit(self, y):
        # Import cvxpy inside method to ensure it's loaded
        try:
            import cvxpy as cp
        except ImportError:
            raise ImportError("cvxpy is required for SJM. Install with: pip install cvxpy")

        T = len(y)
        mu = cp.Variable(T)
        log_sigma = cp.Variable(T)

        resid = y - mu
        loss = cp.sum(cp.square(resid) / cp.exp(2 * log_sigma) + 2 * log_sigma)

        if T > 1:
            first_diff = mu[1:] - mu[:-1]
            persist = self.gamma_ * cp.norm(first_diff, 1)
        else:
            persist = 0

        if T > 2:
            second_diff = mu[2:] - 2 * mu[1:-1] + mu[:-2]
            regime = self.lambda_ * cp.norm(second_diff, 1)
        else:
            regime = 0

        objective = loss + persist + regime
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver=cp.ECOS, verbose=False)

        self.mu_ = mu.value
        self.log_sigma_ = log_sigma.value

        # Detect changepoints
        mu_vals = self.mu_
        diff = np.abs(np.diff(mu_vals))
        threshold = self.threshold * np.std(diff)
        cp_idx = np.where(diff > threshold)[0] + 1
        self.changepoints_ = cp_idx.tolist()

        # Assign regime labels
        unique_mu = np.unique(np.round(mu_vals, decimals=3))
        if len(unique_mu) <= 3:
            regimes = {v: i for i, v in enumerate(sorted(unique_mu))}
            self.regime_labels_ = np.array([regimes[round(v,3)] for v in mu_vals])
        else:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.regime_labels_ = kmeans.fit_predict(mu_vals.reshape(-1,1))

        # Durations
        self.durations_ = []
        start = 0
        for cp_pt in self.changepoints_ + [T]:
            self.durations_.append(cp_pt - start)
            start = cp_pt

        return self

    def get_current_regime(self):
        return int(self.regime_labels_[-1]) if self.regime_labels_ is not None else None

    def get_current_duration(self):
        return self.durations_[-1] if self.durations_ else 0

    def get_transitions(self):
        transitions = []
        for idx in self.changepoints_:
            new_regime = self.regime_labels_[idx] if idx < len(self.regime_labels_) else None
            transitions.append((idx, new_regime))
        return transitions

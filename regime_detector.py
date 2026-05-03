"""
Statistical Jump Model via convex relaxation (fused LASSO + persistence).
Uses cvxpy – ensure it is installed.
"""

import numpy as np
import cvxpy as cp   # <-- must be at top
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
        """
        y : 1D array of observations (e.g., log returns)
        """
        T = len(y)
        # Variables
        mu = cp.Variable(T)
        log_sigma = cp.Variable(T)

        # Negative log-likelihood (Gaussian)
        resid = y - mu
        loss = cp.sum(cp.square(resid) / cp.exp(2 * log_sigma) + 2 * log_sigma)

        # First difference (persistence penalty)
        if T > 1:
            first_diff = mu[1:] - mu[:-1]
            persist = self.gamma_ * cp.norm(first_diff, 1)
        else:
            persist = 0

        # Second difference (regime change penalty) - encourages flat segments
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

        # Detect changepoints where first difference exceeds threshold
        mu_vals = self.mu_
        diff = np.abs(np.diff(mu_vals))
        threshold = self.threshold * np.std(diff)
        cp_idx = np.where(diff > threshold)[0] + 1   # +1 because diff is between points
        self.changepoints_ = cp_idx.tolist()

        # Assign regime labels by clustering the mu values
        unique_mu = np.unique(np.round(mu_vals, decimals=3))
        if len(unique_mu) <= 3:
            regimes = {v: i for i, v in enumerate(sorted(unique_mu))}
            self.regime_labels_ = np.array([regimes[round(v,3)] for v in mu_vals])
        else:
            # cluster into 3 states (low/medium/high)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            self.regime_labels_ = kmeans.fit_predict(mu_vals.reshape(-1,1))

        # Compute persistence (duration of each regime segment)
        self.durations_ = []
        start = 0
        for cp in self.changepoints_ + [T]:
            duration = cp - start
            self.durations_.append(duration)
            start = cp

        return self

    def get_current_regime(self):
        """Return the label of the last regime."""
        if self.regime_labels_ is None:
            return None
        return int(self.regime_labels_[-1])

    def get_current_duration(self):
        """How many days has the current regime lasted?"""
        if not self.durations_:
            return 0
        return self.durations_[-1]

    def get_transitions(self):
        """Return list of (date_index, new_regime) for each changepoint."""
        transitions = []
        for i, idx in enumerate(self.changepoints_):
            new_regime = self.regime_labels_[idx] if idx < len(self.regime_labels_) else None
            transitions.append((idx, new_regime))
        return transitions

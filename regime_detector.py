"""
Statistical Jump Model via convex trend filtering (fused lasso).
Uses rolling volatility to standardise returns, with minimum regime duration.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.cluster import KMeans

class StatisticalJumpModel:
    def __init__(self, lambda_pen=10.0, gamma_persist=20.0, min_regime_days=60,
                 transition_threshold=0.5, vol_window=30):
        self.lambda_ = lambda_pen
        self.gamma_ = gamma_persist
        self.min_regime = min_regime_days
        self.threshold = transition_threshold
        self.vol_window = vol_window
        self.mu_ = None
        self.changepoints_ = None
        self.regime_labels_ = None
        self.durations_ = None

    def fit(self, y):
        """
        y : 1D array of log returns
        """
        # Compute rolling volatility and standardise
        vol = pd.Series(y).rolling(self.vol_window).std().bfill().values
        vol = np.maximum(vol, 1e-6)
        y_std = y / vol

        T = len(y_std)
        mu = cp.Variable(T)

        # Quadratic loss
        loss = cp.sum_squares(y_std - mu)

        # First difference (persistence penalty)
        if T > 1:
            first_diff = mu[1:] - mu[:-1]
            persist = self.gamma_ * cp.norm(first_diff, 1)
        else:
            persist = 0

        # Second difference (regime change penalty)
        if T > 2:
            second_diff = mu[2:] - 2 * mu[1:-1] + mu[:-2]
            regime = self.lambda_ * cp.norm(second_diff, 1)
        else:
            regime = 0

        objective = loss + persist + regime
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver=cp.SCS, verbose=False)

        self.mu_ = mu.value

        # Initial changepoints from first difference
        diff = np.abs(np.diff(self.mu_))
        threshold = self.threshold * np.std(diff)
        cp_idx = np.where(diff > threshold)[0] + 1
        cp_idx = np.unique(np.concatenate(([0], cp_idx, [T])))

        # Enforce minimum regime duration by merging short segments
        merged_cp = [0]
        for i in range(1, len(cp_idx) - 1):
            if cp_idx[i] - merged_cp[-1] >= self.min_regime:
                merged_cp.append(cp_idx[i])
        merged_cp.append(T)
        self.changepoints_ = merged_cp[1:-1]   # exclude 0 and T

        # Assign regime labels by clustering the mu values of each segment
        self.regime_labels_ = np.zeros(T, dtype=int)
        for seg_idx in range(len(merged_cp) - 1):
            start, end = merged_cp[seg_idx], merged_cp[seg_idx+1]
            seg_mu = np.mean(self.mu_[start:end])
            self.regime_labels_[start:end] = seg_idx

        # Cluster segment means into 3 broad regimes (low/medium/high)
        segment_means = [np.mean(self.mu_[merged_cp[i]:merged_cp[i+1]]) for i in range(len(merged_cp)-1)]
        if len(set(segment_means)) <= 3:
            unique = sorted(set(segment_means))
            mapping = {v: i for i, v in enumerate(unique)}
        else:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(np.array(segment_means).reshape(-1,1))
            mapping = {seg_idx: int(labels[i]) for i, seg_idx in enumerate(range(len(segment_means)))}
        # Apply mapping to each time point
        for seg_idx in range(len(segment_means)):
            start, end = merged_cp[seg_idx], merged_cp[seg_idx+1]
            self.regime_labels_[start:end] = mapping[seg_idx]

        # Compute durations (days per regime segment)
        self.durations_ = [merged_cp[i+1] - merged_cp[i] for i in range(len(merged_cp)-1)]

        return self

    def get_current_regime(self):
        return int(self.regime_labels_[-1]) if self.regime_labels_ is not None else None

    def get_current_duration(self):
        return self.durations_[-1] if self.durations_ else 0

    def get_transitions(self):
        """Return list of (date_index, new_regime) for each changepoint."""
        transitions = []
        for idx in self.changepoints_:
            new_regime = self.regime_labels_[idx] if idx < len(self.regime_labels_) else None
            transitions.append((idx, new_regime))
        return transitions

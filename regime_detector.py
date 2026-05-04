"""regime_detector.py — Statistical Jump Model via convex trend filtering."""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


class StatisticalJumpModel:
    """Fit a piecewise-constant trend to log-return series using fused lasso.

    Regimes are labelled 0 (low-return), 1 (mid-return), 2 (high-return)
    based on the *actual* median simple return in each regime — not on the
    standardised trend values — so the regime labels are economically meaningful.
    """

    def __init__(
        self,
        lambda_pen: float = 10.0,
        gamma_persist: float = 20.0,
        min_regime_days: int = 60,
        transition_threshold: float = 0.5,
        vol_window: int = 30,
        n_regimes: int = 3,
    ) -> None:
        self.lambda_ = lambda_pen
        self.gamma_ = gamma_persist
        self.min_regime = min_regime_days
        self.threshold = transition_threshold
        self.vol_window = vol_window
        self.n_regimes = n_regimes

        # Output attributes set by fit()
        self.mu_: np.ndarray | None = None
        self.changepoints_: list[int] = []
        self.regime_labels_: np.ndarray | None = None
        self.segment_boundaries_: list[int] = []
        self.durations_: list[int] = []
        self.regime_return_map_: dict[int, float] = {}  # regime_id → annualised return

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, log_returns: np.ndarray) -> "StatisticalJumpModel":
        """Fit the model to a 1-D array of log returns."""
        y = np.asarray(log_returns, dtype=float)
        T = len(y)

        # Standardise by rolling volatility
        vol = pd.Series(y).rolling(self.vol_window, min_periods=5).std().bfill().values
        vol = np.maximum(vol, 1e-8)
        y_std = y / vol

        # ── Convex optimisation ───────────────────────────────────────────────
        mu = cp.Variable(T)
        loss = cp.sum_squares(y_std - mu)
        persist = self.gamma_ * cp.norm(mu[1:] - mu[:-1], 1) if T > 1 else 0
        smooth = (
            self.lambda_ * cp.norm(mu[2:] - 2 * mu[1:-1] + mu[:-2], 1) if T > 2 else 0
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cp.Problem(cp.Minimize(loss + persist + smooth)).solve(
                solver=cp.SCS, verbose=False, max_iters=5000
            )

        if mu.value is None:
            # Solver failed — assign single regime
            self.mu_ = np.zeros(T)
            self._assign_single_regime(y, T)
            return self

        self.mu_ = mu.value

        # ── Changepoint detection ─────────────────────────────────────────────
        diff = np.abs(np.diff(self.mu_))
        threshold = self.threshold * np.std(diff) if diff.std() > 1e-10 else 0.0
        raw_cp = np.where(diff > threshold)[0] + 1
        boundaries = np.unique(np.concatenate(([0], raw_cp, [T])))

        # Merge segments shorter than min_regime_days
        merged = [0]
        for idx in boundaries[1:-1]:
            if idx - merged[-1] >= self.min_regime:
                merged.append(int(idx))
        merged.append(T)
        self.segment_boundaries_ = merged
        self.changepoints_ = merged[1:-1]

        # ── Regime labelling based on actual returns ───────────────────────────
        simple_returns = np.expm1(y)  # log → simple
        n_segs = len(merged) - 1
        seg_median_returns = np.array(
            [
                np.median(simple_returns[merged[i] : merged[i + 1]])
                for i in range(n_segs)
            ]
        )

        # Cluster segment medians into n_regimes broad regimes
        if n_segs <= self.n_regimes:
            # Not enough segments — rank-order directly
            order = np.argsort(seg_median_returns)
            seg_regime = np.zeros(n_segs, dtype=int)
            for rank, seg_i in enumerate(order):
                seg_regime[seg_i] = min(rank, self.n_regimes - 1)
        else:
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            raw_labels = kmeans.fit_predict(seg_median_returns.reshape(-1, 1))
            # Re-order cluster IDs so 0=low, 1=mid, 2=high by cluster centre
            centres = kmeans.cluster_centers_.flatten()
            order = np.argsort(centres)  # ascending
            remap = {old: new for new, old in enumerate(order)}
            seg_regime = np.array([remap[lbl] for lbl in raw_labels])

        # Assign per-timestep labels
        labels = np.zeros(T, dtype=int)
        for i in range(n_segs):
            labels[merged[i] : merged[i + 1]] = seg_regime[i]
        self.regime_labels_ = labels

        # Durations
        self.durations_ = [merged[i + 1] - merged[i] for i in range(n_segs)]

        # Annualised return per regime.
        # Use 21-day (monthly) rolling compounded returns, then take median across
        # all such windows that fall inside this regime, then annualise × 12.
        # This avoids the explosion that occurs when compounding daily log returns
        # over 252 periods for high-momentum ETFs.
        log_series = pd.Series(y)
        # 21-day compounded log return for each ending day
        rolling_log_21 = log_series.rolling(21).sum()  # sum of log = compound
        rolling_simple_21 = rolling_log_21.apply(np.expm1)  # to simple return

        for reg_id in range(self.n_regimes):
            mask = self.regime_labels_ == reg_id
            if mask.any():
                regime_monthly = rolling_simple_21.values[mask]
                regime_monthly = regime_monthly[np.isfinite(regime_monthly)]
                if len(regime_monthly) > 0:
                    median_monthly = float(np.median(regime_monthly))
                    # Annualise: (1 + r_monthly)^12 - 1
                    ann_return = float((1 + median_monthly) ** 12 - 1)
                    self.regime_return_map_[reg_id] = ann_return
                else:
                    self.regime_return_map_[reg_id] = 0.0
            else:
                self.regime_return_map_[reg_id] = 0.0

        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assign_single_regime(self, y: np.ndarray, T: int) -> None:
        self.regime_labels_ = np.zeros(T, dtype=int)
        self.durations_ = [T]
        rolling_log_21 = pd.Series(y).rolling(21).sum()
        monthly = rolling_log_21.apply(np.expm1).dropna()
        median_monthly = float(np.median(monthly)) if len(monthly) > 0 else 0.0
        ann_simple = float((1 + median_monthly) ** 12 - 1)
        self.regime_return_map_ = {0: ann_simple}

    def get_current_regime(self) -> int:
        return int(self.regime_labels_[-1]) if self.regime_labels_ is not None else 0

    def get_current_duration(self) -> int:
        return int(self.durations_[-1]) if self.durations_ else 0

    def get_transitions(self) -> list[tuple[int, int]]:
        if self.regime_labels_ is None:
            return []
        return [
            (int(idx), int(self.regime_labels_[idx]))
            for idx in self.changepoints_
            if idx < len(self.regime_labels_)
        ]

    def get_regime_name(self, regime_id: int) -> str:
        names = {0: "Low", 1: "Mid", 2: "High"}
        return names.get(regime_id, str(regime_id))

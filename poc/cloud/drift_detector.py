"""
cloud/drift_detector.py

Detects data drift and concept drift in incoming edge data.

Two drift types:
  1. Data Drift    – the input feature distribution has changed
                     (e.g. new sensor, lighting change in factory)
  2. Concept Drift – the relationship between features and labels changed
                     (e.g. a new type of defect that looks different)

Detection method:
  - Kolmogorov-Smirnov (KS) test on each feature dimension
  - Population Stability Index (PSI) as a secondary metric
  - Accuracy monitoring for concept drift

When drift is detected a "pipeline-triggers" Pub/Sub message is published
so the MLOps pipeline can start a retraining run.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple


class DriftDetector:
    """
    Monitors incoming feature distributions against a reference (baseline).

    Parameters
    ----------
    significance : float
        KS test p-value threshold.  p < significance → drift detected.
    window_size  : int
        Number of recent samples to keep in the current window.
    """

    def __init__(self, significance: float = 0.05, window_size: int = 200):
        self.significance = significance
        self.window_size = window_size

        self._reference: Optional[np.ndarray] = None  # [N_ref, D]
        self._current_window: List[np.ndarray] = []   # rolling buffer
        self._drift_events: List[Dict[str, Any]] = []
        self._check_count = 0

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def set_reference(self, X_ref: np.ndarray) -> None:
        """
        Set the reference distribution (training data baseline).
        Called once after the initial model is trained.
        """
        self._reference = X_ref.copy()

    # ------------------------------------------------------------------
    # Update window
    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray) -> None:
        """Add new samples to the sliding window."""
        self._current_window.append(X_new)
        # Trim to window_size
        total = sum(len(b) for b in self._current_window)
        while total > self.window_size and len(self._current_window) > 1:
            removed = self._current_window.pop(0)
            total -= len(removed)

    # ------------------------------------------------------------------
    # KS Test
    # ------------------------------------------------------------------

    def check_data_drift(self) -> Dict[str, Any]:
        """
        Run KS test on each feature dimension.

        Returns a drift report with:
          - is_drift        : bool
          - drifted_features: List[int]
          - ks_statistics   : per-feature KS stat
          - p_values        : per-feature p-values
          - drift_fraction  : fraction of features that drifted
        """
        if self._reference is None or len(self._current_window) == 0:
            return {"is_drift": False, "reason": "insufficient_data"}

        X_cur = np.vstack(self._current_window)
        X_ref = self._reference
        n_features = X_ref.shape[1]

        ks_stats, p_values = [], []
        for d in range(n_features):
            ks, p = stats.ks_2samp(X_ref[:, d], X_cur[:, d])
            ks_stats.append(float(ks))
            p_values.append(float(p))

        drifted = [i for i, p in enumerate(p_values) if p < self.significance]
        drift_fraction = len(drifted) / n_features
        is_drift = drift_fraction > 0.1  # >10% of features drifted

        self._check_count += 1
        result = {
            "check_id": self._check_count,
            "is_drift": is_drift,
            "drift_type": "data_drift",
            "drift_fraction": drift_fraction,
            "drifted_features": drifted[:10],  # report first 10
            "mean_ks_stat": float(np.mean(ks_stats)),
            "min_p_value": float(np.min(p_values)),
            "n_reference": len(X_ref),
            "n_current": len(X_cur),
        }

        if is_drift:
            self._drift_events.append(result)

        return result

    # ------------------------------------------------------------------
    # PSI (Population Stability Index)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """
        Compute PSI for a single feature.
          PSI < 0.1  → no significant change
          PSI < 0.2  → moderate change
          PSI >= 0.2 → significant change
        """
        eps = 1e-9
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        bins = np.linspace(min_val, max_val, n_bins + 1)

        exp_pct = np.histogram(expected, bins=bins)[0] / len(expected) + eps
        act_pct = np.histogram(actual, bins=bins)[0] / len(actual) + eps

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(psi)

    def check_psi(self) -> Dict[str, Any]:
        """Compute mean PSI across all features."""
        if self._reference is None or len(self._current_window) == 0:
            return {"mean_psi": 0.0, "is_drift": False}

        X_cur = np.vstack(self._current_window)
        X_ref = self._reference
        n_features = X_ref.shape[1]

        psi_scores = [
            self.compute_psi(X_ref[:, d], X_cur[:, d])
            for d in range(min(n_features, 10))  # sample 10 features for speed
        ]
        mean_psi = float(np.mean(psi_scores))
        return {
            "mean_psi": mean_psi,
            "is_drift": mean_psi >= 0.2,
            "drift_type": "data_drift_psi",
        }

    # ------------------------------------------------------------------
    # Concept drift (accuracy monitoring)
    # ------------------------------------------------------------------

    def check_concept_drift(
        self,
        recent_accuracy: float,
        baseline_accuracy: float,
        threshold: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Detect concept drift by monitoring accuracy degradation.

        Parameters
        ----------
        recent_accuracy   : model accuracy on recent data
        baseline_accuracy : accuracy at training time
        threshold         : acceptable accuracy drop
        """
        drop = baseline_accuracy - recent_accuracy
        is_drift = drop > threshold
        result = {
            "is_drift": is_drift,
            "drift_type": "concept_drift",
            "recent_accuracy": recent_accuracy,
            "baseline_accuracy": baseline_accuracy,
            "accuracy_drop": drop,
            "threshold": threshold,
        }
        if is_drift:
            self._drift_events.append(result)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "checks_performed": self._check_count,
            "drift_events_detected": len(self._drift_events),
            "reference_size": len(self._reference) if self._reference is not None else 0,
            "current_window_size": sum(len(b) for b in self._current_window),
        }

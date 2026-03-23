"""
cloud/active_learning.py

Implements Active Learning for the regional cloud layer.

In this architecture, labeling data is expensive.  Active Learning selects
only the most *informative* samples for human annotation via
Vertex AI Data Labeling, thereby reducing labeling cost while maintaining
model quality.

Strategies implemented:
  - entropy         : highest entropy samples are most uncertain
  - margin          : smallest gap between top-2 class probabilities
  - least_confident : lowest max probability
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any

from models.defect_detector import DefectDetector


class ActiveLearner:
    """
    Selects the most informative unlabeled samples for annotation.

    Parameters
    ----------
    model : DefectDetector
        The current global model used to estimate sample uncertainty.
    strategy : str
        One of: 'entropy', 'margin', 'least_confident'
    budget : int
        Maximum number of samples to select per round.
    """

    def __init__(
        self,
        model: DefectDetector,
        strategy: str = "entropy",
        budget: int = 20,
    ):
        self.model = model
        self.strategy = strategy
        self.budget = budget
        self._rounds_completed = 0
        self._total_selected = 0

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def select_samples(
        self,
        X_unlabeled: np.ndarray,
        entity_ids: List[str],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Select up to `budget` samples with the highest uncertainty.

        Parameters
        ----------
        X_unlabeled : np.ndarray  [N, input_dim]
        entity_ids  : List[str]  corresponding entity IDs in the feature store

        Returns
        -------
        X_selected      : np.ndarray  [k, input_dim]
        selected_ids    : List[str]
        uncertainty_scores : np.ndarray [k]  (descending)
        """
        if len(X_unlabeled) == 0:
            return np.empty((0, X_unlabeled.shape[-1])), [], np.empty(0)

        x_tensor = torch.from_numpy(X_unlabeled.astype(np.float32))
        scores = self.model.uncertainty(x_tensor, strategy=self.strategy).numpy()

        k = min(self.budget, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        X_selected = X_unlabeled[top_indices]
        selected_ids = [entity_ids[i] for i in top_indices]
        selected_scores = scores[top_indices]

        self._rounds_completed += 1
        self._total_selected += k

        return X_selected, selected_ids, selected_scores

    # ------------------------------------------------------------------
    # Oracle labeling (simulates human annotator / auto-labeling)
    # ------------------------------------------------------------------

    def oracle_label(
        self,
        X: np.ndarray,
        noise: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Simulate human annotation.

        A simple heuristic: defects have higher mean feature values.
        `noise` fraction of labels are flipped to simulate annotator error.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        # Pseudo-ground-truth based on feature mean (matches data generation logic)
        feature_mean = X.mean(axis=-1)
        labels = (feature_mean > 0.5).astype(np.int64)

        # Introduce annotation noise
        flip_mask = rng.random(len(labels)) < noise
        labels[flip_mask] = 1 - labels[flip_mask]

        return labels

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "budget_per_round": self.budget,
            "rounds_completed": self._rounds_completed,
            "total_samples_selected": self._total_selected,
        }

    def update_model(self, model: DefectDetector) -> None:
        """Update the model used for uncertainty estimation."""
        self.model = model

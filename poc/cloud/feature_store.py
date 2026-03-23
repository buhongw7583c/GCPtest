"""
cloud/feature_store.py

Simulates Vertex AI Feature Store.

Key responsibilities:
  - Store and serve features for training / inference consistency
  - Enable cross-factory feature reuse
  - Provide point-in-time correct feature retrieval (avoid training-serving skew)
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple


class FeatureEntity:
    """A single entity (sample) in the feature store."""

    def __init__(
        self,
        entity_id: str,
        features: np.ndarray,
        label: Optional[int],
        factory_id: str,
        timestamp: float,
        source: str = "edge_upload",
    ):
        self.entity_id = entity_id
        self.features = features
        self.label = label
        self.factory_id = factory_id
        self.timestamp = timestamp
        self.source = source
        self.is_labeled = label is not None


class FeatureStore:
    """
    In-memory simulation of Vertex AI Feature Store.

    Supports:
      - Batch ingestion from edge uploads
      - Cross-factory feature retrieval for training
      - Point-in-time correct batch serving
      - Feature statistics (for drift baseline)
    """

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self._store: Dict[str, FeatureEntity] = {}
        self._factory_index: Dict[str, List[str]] = {}  # factory_id → [entity_ids]
        self._counter = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_batch(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray],
        factory_id: str,
        source: str = "edge_upload",
    ) -> List[str]:
        """
        Ingest a batch of feature vectors.

        Returns list of assigned entity_ids.
        """
        entity_ids = []
        labels = y if y is not None else [None] * len(X)
        ts = time.time()

        for feat, label in zip(X, labels):
            eid = f"{factory_id}_{self._counter:06d}"
            self._counter += 1
            entity = FeatureEntity(
                entity_id=eid,
                features=feat.astype(np.float32),
                label=int(label) if label is not None else None,
                factory_id=factory_id,
                timestamp=ts,
                source=source,
            )
            self._store[eid] = entity
            self._factory_index.setdefault(factory_id, []).append(eid)
            entity_ids.append(eid)

        return entity_ids

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_training_data(
        self,
        factory_ids: Optional[List[str]] = None,
        labeled_only: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve features + labels for training.

        Parameters
        ----------
        factory_ids : if None, return data from ALL factories (cross-factory)
        labeled_only : only return samples that have ground-truth labels
        """
        if factory_ids is None:
            entities = list(self._store.values())
        else:
            eids = []
            for fid in factory_ids:
                eids.extend(self._factory_index.get(fid, []))
            entities = [self._store[eid] for eid in eids if eid in self._store]

        if labeled_only:
            entities = [e for e in entities if e.is_labeled]

        if not entities:
            return np.empty((0, self.feature_dim), dtype=np.float32), np.empty(0, dtype=np.int64)

        X = np.stack([e.features for e in entities])
        # Use -1 as sentinel for unlabeled entities when labeled_only=False
        y = np.array([e.label if e.label is not None else -1 for e in entities], dtype=np.int64)
        return X, y

    def get_unlabeled_data(
        self, factory_ids: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Return unlabeled samples (candidates for Active Learning)."""
        if factory_ids is None:
            entities = [e for e in self._store.values() if not e.is_labeled]
        else:
            eids = []
            for fid in factory_ids:
                eids.extend(self._factory_index.get(fid, []))
            entities = [
                self._store[eid] for eid in eids
                if eid in self._store and not self._store[eid].is_labeled
            ]

        if not entities:
            return np.empty((0, self.feature_dim), dtype=np.float32), []

        X = np.stack([e.features for e in entities])
        ids = [e.entity_id for e in entities]
        return X, ids

    def label_entities(self, entity_ids: List[str], labels: List[int]) -> int:
        """Assign labels to entities (simulates human annotation in Vertex AI Data Labeling)."""
        labeled = 0
        for eid, lbl in zip(entity_ids, labels):
            if eid in self._store:
                self._store[eid].label = lbl
                self._store[eid].is_labeled = True
                labeled += 1
        return labeled

    # ------------------------------------------------------------------
    # Feature statistics (used as baseline for drift detection)
    # ------------------------------------------------------------------

    def compute_feature_statistics(self, factory_id: Optional[str] = None) -> Dict[str, Any]:
        """Return per-feature mean/std for drift baseline."""
        X, _ = self.get_training_data(factory_ids=[factory_id] if factory_id else None, labeled_only=False)
        if len(X) == 0:
            return {}
        return {
            "n_samples": len(X),
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0).tolist(),
            "min": X.min(axis=0).tolist(),
            "max": X.max(axis=0).tolist(),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        total = len(self._store)
        labeled = sum(1 for e in self._store.values() if e.is_labeled)
        by_factory = {
            fid: len(eids) for fid, eids in self._factory_index.items()
        }
        return {
            "total_entities": total,
            "labeled": labeled,
            "unlabeled": total - labeled,
            "label_rate": labeled / max(1, total),
            "by_factory": by_factory,
        }

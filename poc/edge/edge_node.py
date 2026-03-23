"""
edge/edge_node.py

Simulates a factory edge node:
  1. Generates / receives sensor / camera data (synthetic feature vectors)
  2. Runs real-time inference (<100 ms target)
  3. Filters which samples to upload to the regional cloud:
       - Defect samples          (always upload)
       - Low-confidence samples  (model is uncertain → useful for retraining)
       - Random subset of normal samples (statistical monitoring)
  4. Provides model update (weights/gradients) to the FL aggregator
"""

import time
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

from models.defect_detector import DefectDetector


class EdgeNode:
    """
    Represents one factory edge node.

    Parameters
    ----------
    factory_id : str
        Unique identifier, e.g. "factory_0"
    model : DefectDetector
        The local inference model (copy of the global model initially)
    mean_shift : float
        Per-factory data distribution shift (simulates real-world variation)
    noise_scale : float
        Per-factory noise level
    confidence_threshold : float
        Samples below this confidence → upload to cloud
    sampling_rate_normal : float
        Fraction of normal-and-confident samples to upload for monitoring
    input_dim : int
        Feature vector dimension
    rng_seed : Optional[int]
        For reproducibility
    """

    def __init__(
        self,
        factory_id: str,
        model: DefectDetector,
        mean_shift: float = 0.0,
        noise_scale: float = 1.0,
        confidence_threshold: float = 0.75,
        sampling_rate_normal: float = 0.05,
        input_dim: int = 64,
        rng_seed: Optional[int] = None,
    ):
        self.factory_id = factory_id
        self.model = model
        self.mean_shift = mean_shift
        self.noise_scale = noise_scale
        self.confidence_threshold = confidence_threshold
        self.sampling_rate_normal = sampling_rate_normal
        self.input_dim = input_dim
        self.rng = np.random.default_rng(rng_seed)

        # Accumulated metrics
        self._total_inferred = 0
        self._defects_detected = 0
        self._uploads = 0

    # ------------------------------------------------------------------
    # Data generation (simulates camera / sensor + edge pre-processing)
    # ------------------------------------------------------------------

    def generate_batch(self, n_samples: int = 100, defect_rate: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a batch of pre-processed sensor readings.

        Returns
        -------
        X : np.ndarray  shape [n_samples, input_dim]
        y : np.ndarray  shape [n_samples]  (0=normal, 1=defect)
        """
        labels = (self.rng.random(n_samples) < defect_rate).astype(np.int64)

        X = self.rng.standard_normal((n_samples, self.input_dim)).astype(np.float32)
        X = X * self.noise_scale + self.mean_shift

        # Defects have a shifted feature distribution (simulates visual anomaly)
        defect_idx = labels == 1
        X[defect_idx] += self.rng.standard_normal((defect_idx.sum(), self.input_dim)).astype(np.float32) * 1.5 + 1.0

        return X, labels

    # ------------------------------------------------------------------
    # Inference (simulates <100ms edge inference)
    # ------------------------------------------------------------------

    def run_inference(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a batch and return results with timing.

        Returns
        -------
        dict with keys: predictions, probabilities, latency_ms
        """
        t0 = time.perf_counter()

        x_tensor = torch.from_numpy(X)
        probs = self.model.predict_proba(x_tensor).numpy()
        preds = probs.argmax(axis=-1)
        latency_ms = (time.perf_counter() - t0) * 1000

        self._total_inferred += len(X)
        self._defects_detected += int(preds.sum())

        return {
            "factory_id": self.factory_id,
            "predictions": preds,
            "probabilities": probs,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # Data filtering (what gets uploaded to regional cloud)
    # ------------------------------------------------------------------

    def filter_for_upload(
        self, X: np.ndarray, y: np.ndarray, inference_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decide which samples to upload based on:
          1. Predicted as defect
          2. Confidence below threshold (uncertain)
          3. Random sample of normal/confident data

        Returns
        -------
        dict with upload_X, upload_y, upload_reason
        """
        probs = inference_result["probabilities"]
        preds = inference_result["predictions"]
        max_conf = probs.max(axis=-1)

        # Masks
        is_defect = preds == 1
        is_uncertain = max_conf < self.confidence_threshold
        is_normal_confident = (~is_defect) & (~is_uncertain)
        random_sample = self.rng.random(len(X)) < self.sampling_rate_normal

        upload_mask = is_defect | is_uncertain | (is_normal_confident & random_sample)

        reasons = []
        for i in range(len(X)):
            if is_defect[i]:
                reasons.append("defect")
            elif is_uncertain[i]:
                reasons.append("uncertain")
            elif is_normal_confident[i] and random_sample[i]:
                reasons.append("monitoring_sample")
            else:
                reasons.append(None)

        upload_reasons = [r for r, m in zip(reasons, upload_mask) if m]
        self._uploads += int(upload_mask.sum())

        return {
            "factory_id": self.factory_id,
            "X": X[upload_mask],
            "y": y[upload_mask],
            "reasons": upload_reasons,
            "n_uploaded": int(upload_mask.sum()),
            "n_total": len(X),
        }

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def update_model(self, global_weights: dict) -> None:
        """Receive updated global model weights from the Control Plane."""
        self.model.set_weights(global_weights)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        defect_rate = self._defects_detected / max(1, self._total_inferred)
        upload_rate = self._uploads / max(1, self._total_inferred)
        return {
            "factory_id": self.factory_id,
            "total_inferred": self._total_inferred,
            "defects_detected": self._defects_detected,
            "defect_rate": defect_rate,
            "total_uploads": self._uploads,
            "upload_rate": upload_rate,
        }

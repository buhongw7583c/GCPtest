"""
models/defect_detector.py

Simple MLP-based defect detection model.
In production this would be a quantized CNN (TensorRT INT8) deployed via
Vertex AI Edge Manager, but for the POC we use a lightweight MLP so the
demo runs without a GPU.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectDetector(nn.Module):
    """
    3-layer MLP for binary defect classification.
    Input  : feature vector of size `input_dim`  (simulates pre-processed
             image features extracted at the edge, e.g. from a CNN backbone)
    Output : logits of shape [batch, num_classes]
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (softmax) – no gradient tracked."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return self.predict_proba(x).argmax(dim=-1)

    def uncertainty(self, x: torch.Tensor, strategy: str = "entropy") -> torch.Tensor:
        """
        Compute per-sample uncertainty score – used by Active Learning.
        strategy:
          'entropy'           – Shannon entropy of the predicted distribution
          'margin'            – 1 - (p_top1 - p_top2)
          'least_confident'   – 1 - p_top1
        """
        probs = self.predict_proba(x)  # [N, C]
        if strategy == "entropy":
            eps = 1e-9
            return -(probs * (probs + eps).log()).sum(dim=-1)
        elif strategy == "margin":
            top2, _ = probs.topk(2, dim=-1)
            return 1.0 - (top2[:, 0] - top2[:, 1])
        else:  # least_confident
            return 1.0 - probs.max(dim=-1).values

    def clone(self) -> "DefectDetector":
        """Return a deep copy of this model."""
        return copy.deepcopy(self)

    def get_weights(self) -> dict:
        """Return state dict (used for federated weight exchange)."""
        return copy.deepcopy(self.state_dict())

    def set_weights(self, state_dict: dict) -> None:
        """Load weights from a state dict."""
        self.load_state_dict(state_dict)

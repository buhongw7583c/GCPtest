"""
edge/local_trainer.py

Performs local federated learning training on each edge node.
In Federated Learning, no raw data leaves the factory – only model
weight updates (gradients / new weights) are sent to the aggregator.

This simulates the "Local Training Node" described in the architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Tuple, Dict, Any

from models.defect_detector import DefectDetector


class LocalTrainer:
    """
    Trains a local model on factory data and returns updated weights.

    Parameters
    ----------
    factory_id : str
    model : DefectDetector
        The model to fine-tune locally (starts from global weights)
    learning_rate : float
    local_epochs : int
        Number of local training epochs before sending weights to aggregator
    batch_size : int
    """

    def __init__(
        self,
        factory_id: str,
        model: DefectDetector,
        learning_rate: float = 0.001,
        local_epochs: int = 3,
        batch_size: int = 32,
    ):
        self.factory_id = factory_id
        self.model = model
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        global_weights: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Run local training.

        Parameters
        ----------
        X : np.ndarray  [N, input_dim]
        y : np.ndarray  [N]  (class labels)
        global_weights : optional state_dict to start from

        Returns
        -------
        dict with keys:
          - weights     : updated state_dict
          - n_samples   : number of training samples (for weighted FedAvg)
          - train_loss  : final epoch loss
          - train_acc   : final epoch accuracy
        """
        if global_weights is not None:
            self.model.set_weights(global_weights)

        # Build dataset
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        final_loss, final_acc = 0.0, 0.0

        for epoch in range(self.local_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(X_batch)
                correct += (logits.argmax(dim=-1) == y_batch).sum().item()
                total += len(X_batch)

            final_loss = epoch_loss / total
            final_acc = correct / total

        self.model.eval()

        return {
            "factory_id": self.factory_id,
            "weights": self.model.get_weights(),
            "n_samples": len(X),
            "train_loss": final_loss,
            "train_acc": final_acc,
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the current model on a held-out set."""
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()
            loss = nn.CrossEntropyLoss()(logits, y_t).item()

        tp = ((preds == 1) & (y_t == 1)).sum().item()
        fp = ((preds == 1) & (y_t == 0)).sum().item()
        fn = ((preds == 0) & (y_t == 1)).sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return {
            "factory_id": self.factory_id,
            "accuracy": acc,
            "loss": loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

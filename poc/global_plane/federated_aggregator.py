"""
global_plane/federated_aggregator.py

Implements Federated Learning aggregation (FedAvg + weighted FedAvg).

In the architecture, this runs in the Global Control Plane and aggregates
model weight updates from multiple factory edge nodes / regional nodes.
EU-compliant: raw data never leaves the regional boundary – only model
weights are exchanged.

References:
  McMahan et al., "Communication-Efficient Learning of Deep Networks from
  Decentralized Data" (FedAvg), 2017
"""

import copy
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from models.defect_detector import DefectDetector


class FederatedAggregator:
    """
    Aggregates local model updates into a single global model.

    Parameters
    ----------
    global_model : DefectDetector
        The current global model (will be updated in-place or cloned).
    aggregation  : str
        'fedavg'          – simple average of all weights
        'weighted_fedavg' – weighted average by number of training samples
    min_clients  : int
        Minimum number of clients required to trigger aggregation.
    """

    def __init__(
        self,
        global_model: DefectDetector,
        aggregation: str = "fedavg",
        min_clients: int = 2,
    ):
        self.global_model = global_model
        self.aggregation = aggregation
        self.min_clients = min_clients
        self._round = 0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate client model updates.

        Parameters
        ----------
        client_updates : list of dicts, each with:
            - factory_id  : str
            - weights     : state_dict (from LocalTrainer.train())
            - n_samples   : int  (number of local training samples)
            - train_loss  : float
            - train_acc   : float

        Returns
        -------
        dict with aggregation results and the new global weights
        """
        if len(client_updates) < self.min_clients:
            return {
                "success": False,
                "reason": f"Not enough clients: {len(client_updates)} < {self.min_clients}",
                "round": self._round,
            }

        self._round += 1

        # Compute per-client weight (fraction of total data)
        total_samples = sum(u["n_samples"] for u in client_updates)
        if self.aggregation == "weighted_fedavg":
            weights_coeff = [u["n_samples"] / total_samples for u in client_updates]
        else:  # fedavg – equal weight
            weights_coeff = [1.0 / len(client_updates)] * len(client_updates)

        # FedAvg: weighted average of all parameter tensors
        new_state = copy.deepcopy(client_updates[0]["weights"])
        for key in new_state:
            new_state[key] = torch.zeros_like(new_state[key], dtype=torch.float32)
            for update, coeff in zip(client_updates, weights_coeff):
                new_state[key] += coeff * update["weights"][key].float()

        self.global_model.set_weights(new_state)

        # Compute aggregated metrics
        avg_loss = np.average(
            [u["train_loss"] for u in client_updates],
            weights=[u["n_samples"] for u in client_updates],
        )
        avg_acc = np.average(
            [u["train_acc"] for u in client_updates],
            weights=[u["n_samples"] for u in client_updates],
        )

        result = {
            "success": True,
            "round": self._round,
            "aggregation": self.aggregation,
            "n_clients": len(client_updates),
            "total_samples": total_samples,
            "avg_train_loss": float(avg_loss),
            "avg_train_acc": float(avg_acc),
            "client_contributions": [
                {
                    "factory_id": u["factory_id"],
                    "n_samples": u["n_samples"],
                    "weight": round(w, 4),
                    "train_acc": round(u["train_acc"], 4),
                }
                for u, w in zip(client_updates, weights_coeff)
            ],
        }

        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_global_weights(self) -> dict:
        return self.global_model.get_weights()

    def get_history(self) -> List[Dict[str, Any]]:
        return self._history

    @property
    def current_round(self) -> int:
        return self._round

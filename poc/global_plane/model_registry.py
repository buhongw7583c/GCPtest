"""
global_plane/model_registry.py

Simulates Vertex AI Model Registry + Artifact Registry.

Responsibilities:
  - Track all model versions with metadata
  - Support canary / A-B / shadow deployment strategies
  - Manage promotion lifecycle: staging → production
  - Store evaluation metrics per version
"""

import time
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from models.defect_detector import DefectDetector


@dataclass
class ModelVersion:
    """Metadata for a single model version."""
    version_id: str
    model: DefectDetector
    created_at: float = field(default_factory=time.time)
    fl_round: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    stage: str = "staging"          # staging | production | archived
    deployment_strategy: str = "none"  # none | canary | ab_test | shadow
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "fl_round": self.fl_round,
            "metrics": self.metrics,
            "tags": self.tags,
            "stage": self.stage,
            "deployment_strategy": self.deployment_strategy,
            "description": self.description,
        }


class ModelRegistry:
    """
    In-memory simulation of Vertex AI Model Registry.

    Supports:
      - Registering new model versions after each FL round
      - Promoting models from staging → production
      - Canary, A/B, and Shadow deployment strategies
      - Version rollback
    """

    def __init__(self, max_versions: int = 10):
        self.max_versions = max_versions
        self._versions: Dict[str, ModelVersion] = {}
        self._production_version: Optional[str] = None
        self._version_counter = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model: DefectDetector,
        fl_round: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> str:
        """
        Register a new model version.

        Returns the version_id.
        """
        self._version_counter += 1
        version_id = f"v{self._version_counter}.0"

        mv = ModelVersion(
            version_id=version_id,
            model=copy.deepcopy(model),
            fl_round=fl_round,
            metrics=metrics or {},
            tags=tags or {},
            description=description,
        )
        self._versions[version_id] = mv

        # Prune old archived versions if over limit
        self._prune()

        return version_id

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def promote_to_production(
        self,
        version_id: str,
        strategy: str = "canary",
    ) -> Dict[str, Any]:
        """
        Promote a version to production.

        strategy:
          'canary'  – gradual rollout to a small % of edge nodes first
          'ab_test' – split traffic between old and new model
          'shadow'  – new model runs in parallel (shadow mode), no live impact
          'direct'  – immediate full deployment
        """
        if version_id not in self._versions:
            return {"success": False, "reason": f"Version {version_id} not found"}

        # Archive current production
        if self._production_version and self._production_version in self._versions:
            self._versions[self._production_version].stage = "archived"

        mv = self._versions[version_id]
        mv.stage = "production"
        mv.deployment_strategy = strategy
        self._production_version = version_id

        return {
            "success": True,
            "version_id": version_id,
            "strategy": strategy,
            "previous_production": self._production_version,
        }

    def get_production_model(self) -> Optional[DefectDetector]:
        """Return the current production model."""
        if self._production_version is None:
            return None
        mv = self._versions.get(self._production_version)
        return mv.model if mv else None

    def get_production_version_id(self) -> Optional[str]:
        return self._production_version

    def rollback(self) -> Dict[str, Any]:
        """Roll back to the previous production version."""
        archived = [
            v for v in self._versions.values()
            if v.stage == "archived"
        ]
        if not archived:
            return {"success": False, "reason": "No archived version to roll back to"}

        # Find most recently archived
        prev = max(archived, key=lambda v: v.created_at)

        # Archive current production
        if self._production_version and self._production_version in self._versions:
            self._versions[self._production_version].stage = "archived"

        prev.stage = "production"
        self._production_version = prev.version_id
        return {"success": True, "rolled_back_to": prev.version_id}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        return self._versions.get(version_id)

    def list_versions(self) -> List[Dict[str, Any]]:
        return [v.to_dict() for v in sorted(self._versions.values(), key=lambda v: v.created_at)]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove oldest archived versions if over the limit."""
        archived = sorted(
            [v for v in self._versions.values() if v.stage == "archived"],
            key=lambda v: v.created_at,
        )
        while len(self._versions) > self.max_versions and archived:
            oldest = archived.pop(0)
            del self._versions[oldest.version_id]

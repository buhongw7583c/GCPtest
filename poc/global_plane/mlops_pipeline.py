"""
global_plane/mlops_pipeline.py

Simulates Vertex AI Pipelines – the MLOps CI/CD/CT orchestration layer.

Responsibilities:
  - Manage the full ML lifecycle: data → features → train → evaluate → deploy
  - Auto-trigger retraining on: drift, performance drop, new data, schedule
  - Coordinate all components across edge, regional cloud, and global plane
  - Emit pipeline-trigger Pub/Sub events
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum


class TriggerReason(Enum):
    SCHEDULED = "scheduled"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    NEW_DATA = "new_data"
    PERFORMANCE_DROP = "performance_drop"
    MANUAL = "manual"


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    name: str
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


@dataclass
class PipelineRun:
    run_id: str
    trigger_reason: TriggerReason
    fl_round: int
    steps: List[PipelineStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: PipelineStatus = PipelineStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trigger_reason": self.trigger_reason.value,
            "fl_round": self.fl_round,
            "status": self.status.value,
            "duration_ms": (self.end_time - self.start_time) * 1000 if self.end_time else None,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "duration_ms": round(s.duration_ms, 1),
                    "output": s.output,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }


class MLOpsPipeline:
    """
    Orchestrates the end-to-end MLOps pipeline.

    Pipeline steps (matching the architecture):
      1. data_validation      – check data quality from feature store
      2. feature_engineering  – compute / refresh features
      3. federated_training   – run FL round across all factories
      4. model_evaluation     – evaluate on held-out data
      5. model_registration   – register in model registry
      6. deployment           – promote to production with chosen strategy
      7. monitoring_setup     – configure drift & performance monitoring

    Parameters
    ----------
    drift_trigger_threshold    : float  p-value threshold
    performance_drop_threshold : float  accuracy drop to trigger retrain
    """

    def __init__(
        self,
        drift_trigger_threshold: float = 0.05,
        performance_drop_threshold: float = 0.03,
    ):
        self.drift_trigger_threshold = drift_trigger_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self._runs: List[PipelineRun] = []
        self._run_counter = 0
        self._baseline_accuracy: Optional[float] = None

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def should_trigger(
        self,
        drift_result: Optional[Dict[str, Any]] = None,
        recent_accuracy: Optional[float] = None,
        force: bool = False,
    ) -> Optional[TriggerReason]:
        """
        Decide whether to trigger a new pipeline run.
        Returns TriggerReason or None.
        """
        if force:
            return TriggerReason.MANUAL

        if drift_result:
            if drift_result.get("is_drift"):
                drift_type = drift_result.get("drift_type", "data_drift")
                return TriggerReason.DATA_DRIFT if "psi" not in drift_type else TriggerReason.DATA_DRIFT

        if recent_accuracy is not None and self._baseline_accuracy is not None:
            drop = self._baseline_accuracy - recent_accuracy
            if drop > self.performance_drop_threshold:
                return TriggerReason.PERFORMANCE_DROP

        return None

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(
        self,
        trigger_reason: TriggerReason,
        fl_round: int,
        step_executors: Dict[str, Callable[[], Dict[str, Any]]],
    ) -> PipelineRun:
        """
        Execute the pipeline.

        Parameters
        ----------
        trigger_reason  : why the pipeline was triggered
        fl_round        : current FL round number
        step_executors  : dict mapping step_name → callable that returns output dict
                          The callables are provided by main.py and can call the
                          actual components (trainer, aggregator, etc.)
        """
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}"

        pipeline_steps = [
            "data_validation",
            "feature_engineering",
            "federated_training",
            "model_evaluation",
            "model_registration",
            "deployment",
            "monitoring_setup",
        ]

        pr = PipelineRun(
            run_id=run_id,
            trigger_reason=trigger_reason,
            fl_round=fl_round,
        )
        pr.steps = [PipelineStep(name=s) for s in pipeline_steps]
        pr.status = PipelineStatus.RUNNING
        self._runs.append(pr)

        for step in pr.steps:
            step.start_time = time.time()
            step.status = PipelineStatus.RUNNING
            try:
                executor = step_executors.get(step.name)
                if executor is not None:
                    step.output = executor() or {}
                else:
                    step.output = {"skipped": True}
                step.status = PipelineStatus.SUCCEEDED
            except Exception as exc:
                step.status = PipelineStatus.FAILED
                step.error = str(exc)
                pr.status = PipelineStatus.FAILED
                pr.end_time = time.time()
                return pr
            finally:
                step.end_time = time.time()

        pr.status = PipelineStatus.SUCCEEDED
        pr.end_time = time.time()
        return pr

    # ------------------------------------------------------------------
    # Baseline tracking
    # ------------------------------------------------------------------

    def set_baseline_accuracy(self, accuracy: float) -> None:
        self._baseline_accuracy = accuracy

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def list_runs(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._runs]

    @property
    def total_runs(self) -> int:
        return len(self._runs)

"""
pipeline_state.py – Thread-safe pipeline state and event store.

Captures structured events from every pipeline component so that the
Streamlit UI can render them in real time and the Agent can query them.
"""

import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PipelineEvent:
    """Single event emitted during pipeline execution."""
    timestamp: float
    round: int
    layer: str          # "edge" | "cloud" | "global" | "system"
    component: str      # e.g. "EdgeNode", "DriftDetector", "ModelRegistry"
    event_type: str     # e.g. "inference", "drift_check", "model_registered"
    message: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineState:
    """
    Thread-safe store for pipeline events and component snapshots.

    The pipeline runner writes events here; the UI reads them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[PipelineEvent] = []
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        self._current_round: int = -1
        self._status: str = "idle"          # idle | running | completed | error
        self._error: Optional[str] = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    # ── writing (called by pipeline_runner) ──────────────────────────────

    def emit(
        self,
        round: int,
        layer: str,
        component: str,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        event = PipelineEvent(
            timestamp=time.time(),
            round=round,
            layer=layer,
            component=component,
            event_type=event_type,
            message=message,
            data=data or {},
        )
        with self._lock:
            self._events.append(event)

    def set_snapshot(self, key: str, snapshot: Dict[str, Any]) -> None:
        with self._lock:
            self._snapshots[key] = snapshot

    def set_round(self, round_num: int) -> None:
        with self._lock:
            self._current_round = round_num

    def set_status(self, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            self._status = status
            if status == "running" and self._start_time is None:
                self._start_time = time.time()
            if status in ("completed", "error"):
                self._end_time = time.time()
            if error:
                self._error = error

    # ── reading (called by UI / agent) ───────────────────────────────────

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def current_round(self) -> int:
        with self._lock:
            return self._current_round

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._error

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._start_time is None:
                return 0.0
            end = self._end_time or time.time()
            return end - self._start_time

    def get_events(
        self,
        since: float = 0.0,
        layer: Optional[str] = None,
        component: Optional[str] = None,
        round_num: Optional[int] = None,
    ) -> List[PipelineEvent]:
        with self._lock:
            out = list(self._events)
        if since > 0:
            out = [e for e in out if e.timestamp > since]
        if layer:
            out = [e for e in out if e.layer == layer]
        if component:
            out = [e for e in out if e.component == component]
        if round_num is not None:
            out = [e for e in out if e.round == round_num]
        return out

    def get_snapshot(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._snapshots.get(key)

    def get_all_snapshots(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._snapshots)

    def event_count(self) -> int:
        with self._lock:
            return len(self._events)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._snapshots.clear()
            self._current_round = -1
            self._status = "idle"
            self._error = None
            self._start_time = None
            self._end_time = None

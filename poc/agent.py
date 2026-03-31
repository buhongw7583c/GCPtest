"""
agent.py – Natural-language query agent for the pipeline state.

Parses user questions with keyword matching and returns structured
answers drawn from PipelineState snapshots and events.
"""

from typing import Any, Dict, List, Optional
from pipeline_state import PipelineState


# ──────────────────────────────────────────────────────────────────────────────
# Intent detection helpers
# ──────────────────────────────────────────────────────────────────────────────

_INTENTS = {
    "status": [
        "status", "state", "running", "progress", "how is", "what round",
        "current round", "pipeline status", "is it done", "finished",
    ],
    "model": [
        "model", "version", "registry", "production", "deployed",
        "staging", "accuracy", "which model", "current model",
    ],
    "drift": [
        "drift", "ks test", "psi", "distribution", "shift",
        "concept drift", "data drift",
    ],
    "edge": [
        "edge", "factory", "inference", "latency", "defect",
        "upload", "node",
    ],
    "feature_store": [
        "feature store", "feature_store", "entities", "labeled",
        "unlabeled", "label rate",
    ],
    "active_learning": [
        "active learning", "active_learning", "uncertainty",
        "annotation", "labeling", "selected samples",
    ],
    "pubsub": [
        "pubsub", "pub/sub", "topic", "messages", "queue",
        "published",
    ],
    "pipeline_runs": [
        "pipeline run", "mlops", "ci/cd", "steps", "pipeline step",
        "trigger", "ci cd",
    ],
    "aggregator": [
        "aggregat", "fedavg", "federated", "global model",
        "fl round",
    ],
    "events": [
        "event", "log", "history", "what happened", "show me",
        "tell me everything", "summary", "all",
    ],
    "help": [
        "help", "what can", "how to", "commands", "capabilities",
    ],
}


def _detect_intent(question: str) -> str:
    q = question.lower()
    best, best_score = "events", 0
    for intent, keywords in _INTENTS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best, best_score = intent, score
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Response builders
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_status(state: PipelineState) -> str:
    lines = [
        f"**Pipeline status:** `{state.status}`",
        f"**Current round:** {state.current_round}",
        f"**Elapsed:** {state.elapsed_seconds:.1f}s",
        f"**Total events:** {state.event_count()}",
    ]
    if state.error:
        lines.append(f"**Error:** {state.error}")
    return "\n".join(lines)


def _fmt_model(state: PipelineState) -> str:
    snap = state.get_snapshot("model_registry")
    if not snap:
        return "Model registry data not yet available – pipeline may not have started."
    lines = [f"**Production model:** `{snap.get('production_version', 'N/A')}`\n"]
    lines.append("| Version | Stage | FL Round | Accuracy | Strategy | Description |")
    lines.append("|---------|-------|----------|----------|----------|-------------|")
    for v in snap.get("versions", []):
        acc = v.get("metrics", {}).get("accuracy", 0)
        lines.append(
            f"| {v['version_id']} | {v['stage']} | {v['fl_round']} | "
            f"{acc:.4f} | {v.get('deployment_strategy', '-')} | "
            f"{v.get('description', '')[:40]} |"
        )
    return "\n".join(lines)


def _fmt_drift(state: PipelineState) -> str:
    snap = state.get_snapshot("drift_detector")
    events = state.get_events(component="DriftDetector")
    lines = []
    if snap:
        lines.append(
            f"**Checks performed:** {snap['checks_performed']}  "
            f"**Drift events detected:** {snap['drift_events_detected']}\n"
            f"**Reference size:** {snap['reference_size']}  "
            f"**Current window:** {snap['current_window_size']}"
        )
    if events:
        lines.append("\n**Recent drift events:**")
        for e in events[-5:]:
            lines.append(f"- Round {e.round}: {e.message}")
    if not lines:
        return "No drift data available yet."
    return "\n".join(lines)


def _fmt_edge(state: PipelineState) -> str:
    snap = state.get_snapshot("edge_nodes")
    if not snap:
        return "Edge node data not yet available."
    lines = ["| Factory | Inferred | Defects | Defect Rate | Uploads | Upload Rate |",
             "|---------|----------|---------|-------------|---------|-------------|"]
    for fid, s in snap.items():
        lines.append(
            f"| {fid} | {s['total_inferred']} | {s['defects_detected']} | "
            f"{s['defect_rate']:.2%} | {s['total_uploads']} | "
            f"{s['upload_rate']:.2%} |"
        )
    return "\n".join(lines)


def _fmt_feature_store(state: PipelineState) -> str:
    snap = state.get_snapshot("feature_store")
    if not snap:
        return "Feature store data not yet available."
    lines = [
        f"**Total entities:** {snap['total_entities']}",
        f"**Labeled:** {snap['labeled']}  **Unlabeled:** {snap['unlabeled']}",
        f"**Label rate:** {snap['label_rate']:.2%}",
        f"**By factory:** {snap.get('by_factory', {})}",
    ]
    return "\n".join(lines)


def _fmt_active_learning(state: PipelineState) -> str:
    snap = state.get_snapshot("active_learner")
    if not snap:
        return "Active learning data not yet available."
    lines = [
        f"**Strategy:** {snap['strategy']}",
        f"**Budget per round:** {snap['budget_per_round']}",
        f"**Rounds completed:** {snap['rounds_completed']}",
        f"**Total samples selected:** {snap['total_samples_selected']}",
    ]
    return "\n".join(lines)


def _fmt_pubsub(state: PipelineState) -> str:
    snap = state.get_snapshot("pubsub")
    if not snap:
        return "Pub/Sub data not yet available."
    lines = ["| Topic | Published | Queue Size |",
             "|-------|-----------|------------|"]
    for topic, s in snap.items():
        lines.append(f"| {topic} | {s['total_published']} | {s['queue_size']} |")
    return "\n".join(lines)


def _fmt_pipeline_runs(state: PipelineState) -> str:
    snap = state.get_snapshot("mlops_pipeline")
    if not snap:
        return "MLOps pipeline data not yet available."
    lines = [f"**Total runs:** {snap.get('total_runs', 0)}\n"]
    for run in snap.get("runs", []):
        lines.append(
            f"**{run['run_id']}** – trigger: `{run['trigger_reason']}`  "
            f"status: `{run['status']}`  duration: {run.get('duration_ms', 0):.0f}ms"
        )
        for step in run.get("steps", []):
            icon = "✅" if step["status"] == "succeeded" else "❌"
            lines.append(f"  {icon} {step['name']}  ({step['duration_ms']:.0f}ms)")
    return "\n".join(lines)


def _fmt_aggregator(state: PipelineState) -> str:
    snap = state.get_snapshot("aggregator")
    if not snap:
        return "Aggregator data not yet available."
    lines = [f"**Current FL round:** {snap['current_round']}\n"]
    lines.append("| Round | Clients | Samples | Avg Loss | Avg Acc |")
    lines.append("|-------|---------|---------|----------|---------|")
    for h in snap.get("history", []):
        lines.append(
            f"| {h.get('round', '-')} | {h.get('n_clients', '-')} | "
            f"{h.get('total_samples', '-')} | {h.get('avg_train_loss', 0):.4f} | "
            f"{h.get('avg_train_acc', 0):.4f} |"
        )
    return "\n".join(lines)


def _fmt_events(state: PipelineState, round_num: Optional[int] = None) -> str:
    events = state.get_events(round_num=round_num)
    if not events:
        return "No events recorded yet."
    lines = [f"**Showing {len(events)} events" +
             (f" (round {round_num})" if round_num is not None else "") + ":**\n"]
    for e in events[-30:]:  # last 30 to keep response manageable
        lines.append(f"- `[R{e.round}]` **{e.component}** ({e.layer}): {e.message}")
    if len(events) > 30:
        lines.insert(1, f"_(showing last 30 of {len(events)})_\n")
    return "\n".join(lines)


def _fmt_help() -> str:
    return (
        "I can answer questions about the manufacturing AI pipeline.\n\n"
        "**Try asking about:**\n"
        "- **Pipeline status** – _\"What is the pipeline status?\"_\n"
        "- **Model versions** – _\"Which model is in production?\"_\n"
        "- **Drift detection** – _\"Has any drift been detected?\"_\n"
        "- **Edge nodes** – _\"Show me edge node statistics\"_\n"
        "- **Feature Store** – _\"How many entities are in the feature store?\"_\n"
        "- **Active Learning** – _\"How many samples were selected for labeling?\"_\n"
        "- **Pub/Sub** – _\"Show Pub/Sub message counts\"_\n"
        "- **MLOps Pipeline** – _\"What pipeline runs have executed?\"_\n"
        "- **FL Aggregation** – _\"Show federated aggregation history\"_\n"
        "- **Event log** – _\"What happened in round 2?\"_\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def ask(question: str, state: PipelineState) -> str:
    """
    Answer a natural-language *question* using current pipeline *state*.

    Returns a Markdown-formatted answer string.
    """
    intent = _detect_intent(question)

    # Check for round-specific event queries
    q_lower = question.lower()
    round_num = None
    for r in range(10):
        if f"round {r}" in q_lower:
            round_num = r
            break

    handlers = {
        "status": lambda: _fmt_status(state),
        "model": lambda: _fmt_model(state),
        "drift": lambda: _fmt_drift(state),
        "edge": lambda: _fmt_edge(state),
        "feature_store": lambda: _fmt_feature_store(state),
        "active_learning": lambda: _fmt_active_learning(state),
        "pubsub": lambda: _fmt_pubsub(state),
        "pipeline_runs": lambda: _fmt_pipeline_runs(state),
        "aggregator": lambda: _fmt_aggregator(state),
        "events": lambda: _fmt_events(state, round_num=round_num),
        "help": _fmt_help,
    }

    return handlers.get(intent, lambda: _fmt_events(state, round_num=round_num))()

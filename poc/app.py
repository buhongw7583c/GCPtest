"""
app.py – Streamlit UI for real-time pipeline monitoring and agent queries.

Run with:
    cd poc/
    streamlit run app.py
"""

import sys
import os
import threading
import time

import pandas as pd
import streamlit as st

# Ensure poc/ packages are importable
sys.path.insert(0, os.path.dirname(__file__))

from pipeline_state import PipelineState
from pipeline_runner import run_pipeline
from agent import ask

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Manufacturing AI Pipeline",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────

if "pipeline_state" not in st.session_state:
    st.session_state.pipeline_state = PipelineState()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pipeline_thread" not in st.session_state:
    st.session_state.pipeline_thread = None

ps: PipelineState = st.session_state.pipeline_state

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – controls & component selector
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏭 Manufacturing AI")
    st.caption("Federated Learning + Edge + MLOps on GCP")

    st.divider()

    # Pipeline controls
    st.subheader("Pipeline Controls")
    col1, col2 = st.columns(2)
    with col1:
        start_disabled = ps.status in ("running",)
        if st.button("▶️ Start Pipeline", disabled=start_disabled, use_container_width=True):
            ps.reset()
            st.session_state.chat_history = []

            def _run():
                run_pipeline(ps)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            st.session_state.pipeline_thread = t
            st.rerun()

    with col2:
        reset_disabled = ps.status == "running"
        if st.button("🔄 Reset", disabled=reset_disabled, use_container_width=True):
            ps.reset()
            st.session_state.chat_history = []
            st.rerun()

    # Status indicator
    status_emoji = {
        "idle": "⏸️", "running": "🔄", "completed": "✅", "error": "❌",
    }
    st.metric("Status", f"{status_emoji.get(ps.status, '❓')} {ps.status.upper()}")
    st.metric("Current Round", ps.current_round if ps.current_round >= 0 else "—")
    st.metric("Events", ps.event_count())
    st.metric("Elapsed", f"{ps.elapsed_seconds:.1f}s")

    st.divider()

    # Component quick-nav
    st.subheader("Quick View")
    view_option = st.radio(
        "Select component",
        ["📋 Event Feed", "🏭 Edge Nodes", "🌍 Feature Store",
         "🌍 Drift Detector", "🌍 Active Learning", "🔁 Pub/Sub",
         "🧠 FL Aggregator", "🧠 Model Registry", "🧠 MLOps Pipeline"],
        index=0,
    )

    st.divider()
    st.subheader("Auto-Refresh")
    auto_refresh = st.checkbox("Enable (2s)", value=ps.status == "running")

# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────

st.title("🚀 Pipeline Dashboard")

# ── Top metrics row
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    prod_ver = "—"
    reg_snap = ps.get_snapshot("model_registry")
    if reg_snap:
        prod_ver = reg_snap.get("production_version", "—") or "—"
    st.metric("Production Model", prod_ver)
with m2:
    dd_snap = ps.get_snapshot("drift_detector")
    drift_events = dd_snap["drift_events_detected"] if dd_snap else 0
    st.metric("Drift Events", drift_events)
with m3:
    fs_snap = ps.get_snapshot("feature_store")
    entities = fs_snap["total_entities"] if fs_snap else 0
    st.metric("Feature Store Entities", entities)
with m4:
    al_snap = ps.get_snapshot("active_learner")
    al_selected = al_snap["total_samples_selected"] if al_snap else 0
    st.metric("AL Samples Selected", al_selected)
with m5:
    agg_snap = ps.get_snapshot("aggregator")
    fl_round = agg_snap["current_round"] if agg_snap else 0
    st.metric("FL Rounds", fl_round)

st.divider()

# ── Component detail panels
tab_detail, tab_agent = st.tabs(["📊 Component Detail", "🤖 Agent Query"])

with tab_detail:
    if "Event Feed" in view_option:
        st.subheader("📋 Real-Time Event Feed")
        round_filter = st.selectbox(
            "Filter by round", ["All"] + [str(r) for r in range(4)], index=0,
        )
        rn = None if round_filter == "All" else int(round_filter)
        events = ps.get_events(round_num=rn)
        if events:
            for e in reversed(events[-50:]):
                layer_icon = {"edge": "🏭", "cloud": "🌍", "global": "🧠", "system": "⚙️"}.get(e.layer, "")
                st.markdown(
                    f"`R{e.round}` {layer_icon} **{e.component}** · {e.message}"
                )
        else:
            st.info("No events yet – start the pipeline to see results.")

    elif "Edge Nodes" in view_option:
        st.subheader("🏭 Edge Node Statistics")
        snap = ps.get_snapshot("edge_nodes")
        if snap:
            cols = st.columns(len(snap))
            for col, (fid, s) in zip(cols, snap.items()):
                with col:
                    st.markdown(f"**{fid}**")
                    st.metric("Inferred", s["total_inferred"])
                    st.metric("Defects", s["defects_detected"])
                    st.metric("Defect Rate", f"{s['defect_rate']:.1%}")
                    st.metric("Upload Rate", f"{s['upload_rate']:.1%}")
        else:
            st.info("Edge node data not yet available.")

    elif "Feature Store" in view_option:
        st.subheader("🌍 Feature Store")
        snap = ps.get_snapshot("feature_store")
        if snap:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Entities", snap["total_entities"])
            c2.metric("Labeled", snap["labeled"])
            c3.metric("Unlabeled", snap["unlabeled"])
            c4.metric("Label Rate", f"{snap['label_rate']:.1%}")
            st.json(snap.get("by_factory", {}))
        else:
            st.info("Feature Store data not yet available.")

    elif "Drift Detector" in view_option:
        st.subheader("🌍 Drift Detector")
        snap = ps.get_snapshot("drift_detector")
        if snap:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Checks", snap["checks_performed"])
            c2.metric("Drift Events", snap["drift_events_detected"])
            c3.metric("Reference Size", snap["reference_size"])
            c4.metric("Window Size", snap["current_window_size"])
        else:
            st.info("Drift detector data not yet available.")

        drift_events = ps.get_events(component="DriftDetector")
        if drift_events:
            st.markdown("**Drift Check History:**")
            for e in drift_events:
                st.markdown(f"- `R{e.round}` {e.message}")

    elif "Active Learning" in view_option:
        st.subheader("🌍 Active Learning")
        snap = ps.get_snapshot("active_learner")
        if snap:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Strategy", snap["strategy"])
            c2.metric("Budget/Round", snap["budget_per_round"])
            c3.metric("Rounds", snap["rounds_completed"])
            c4.metric("Total Selected", snap["total_samples_selected"])
        else:
            st.info("Active learning data not yet available.")

    elif "Pub/Sub" in view_option:
        st.subheader("🔁 Pub/Sub Topics")
        snap = ps.get_snapshot("pubsub")
        if snap:
            for topic, s in snap.items():
                if s["total_published"] > 0:
                    st.markdown(f"**{topic}**: {s['total_published']} published, "
                                f"{s['queue_size']} queued")
        else:
            st.info("Pub/Sub data not yet available.")

    elif "FL Aggregator" in view_option:
        st.subheader("🧠 Federated Aggregator")
        snap = ps.get_snapshot("aggregator")
        if snap:
            st.metric("Current FL Round", snap["current_round"])
            if snap.get("history"):
                st.markdown("**Aggregation History:**")
                for h in snap["history"]:
                    st.markdown(
                        f"- Round {h.get('round', '?')}: "
                        f"{h.get('n_clients', '?')} clients, "
                        f"{h.get('total_samples', '?')} samples, "
                        f"avg_acc={h.get('avg_train_acc', 0):.4f}"
                    )
        else:
            st.info("Aggregator data not yet available.")

    elif "Model Registry" in view_option:
        st.subheader("🧠 Model Registry")
        snap = ps.get_snapshot("model_registry")
        if snap:
            st.metric("Production Version", snap.get("production_version", "—"))
            versions = snap.get("versions", [])
            if versions:
                df = pd.DataFrame([
                    {
                        "Version": v["version_id"],
                        "Stage": v["stage"],
                        "FL Round": v["fl_round"],
                        "Accuracy": v.get("metrics", {}).get("accuracy", 0),
                        "Strategy": v.get("deployment_strategy", "—"),
                        "Description": v.get("description", "")[:50],
                    }
                    for v in versions
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Model Registry data not yet available.")

    elif "MLOps Pipeline" in view_option:
        st.subheader("🧠 MLOps Pipeline Runs")
        snap = ps.get_snapshot("mlops_pipeline")
        if snap:
            st.metric("Total Runs", snap.get("total_runs", 0))
            for run in snap.get("runs", []):
                with st.expander(
                    f"{run['run_id']} – {run['trigger_reason']} ({run['status']})"
                ):
                    for step in run.get("steps", []):
                        icon = "✅" if step["status"] == "succeeded" else "❌"
                        st.markdown(
                            f"{icon} **{step['name']}** – {step['duration_ms']:.0f}ms"
                        )
                        if step.get("output"):
                            st.json(step["output"])
        else:
            st.info("MLOps pipeline data not yet available.")


# ── Agent chat interface
with tab_agent:
    st.subheader("🤖 Pipeline Agent")
    st.caption(
        "Ask questions about the pipeline in natural language. "
        "Try: *\"What is the pipeline status?\"*, *\"Which model is in production?\"*, "
        "*\"Has drift been detected?\"*"
    )

    # Render existing conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about the pipeline…"):
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        answer = ask(user_input, ps)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

# ──────────────────────────────────────────────────────────────────────────────
# Auto-refresh
# ──────────────────────────────────────────────────────────────────────────────

if auto_refresh and ps.status == "running":
    time.sleep(2)
    st.rerun()

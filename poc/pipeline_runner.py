"""
pipeline_runner.py – Run the manufacturing AI pipeline with event emission.

Re-uses the same component setup and round logic as main.py, but emits
structured events to a PipelineState so the Streamlit UI can render
progress in real time.
"""

import os
import sys
import time
import numpy as np
import yaml
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(__file__))

from models.defect_detector import DefectDetector
from edge.edge_node import EdgeNode
from edge.local_trainer import LocalTrainer
from cloud.pubsub_simulator import PubSubSimulator
from cloud.feature_store import FeatureStore
from cloud.active_learning import ActiveLearner
from cloud.drift_detector import DriftDetector
from global_plane.federated_aggregator import FederatedAggregator
from global_plane.model_registry import ModelRegistry
from global_plane.mlops_pipeline import MLOpsPipeline, TriggerReason
from pipeline_state import PipelineState


def _load_config() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_components(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate all POC components (same as main.setup_components)."""
    m_cfg = cfg["model"]
    e_cfg = cfg["edge"]
    c_cfg = cfg["cloud"]
    g_cfg = cfg["global"]
    d_cfg = cfg["data"]

    global_model = DefectDetector(
        input_dim=m_cfg["input_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_classes=m_cfg["num_classes"],
    )

    factory_dists = d_cfg["factory_distributions"]
    edge_nodes, local_trainers = [], []
    for i in range(e_cfg["num_factories"]):
        fid = f"factory_{i}"
        dist = factory_dists[fid]
        local_model = DefectDetector(
            input_dim=m_cfg["input_dim"],
            hidden_dim=m_cfg["hidden_dim"],
            num_classes=m_cfg["num_classes"],
        )
        local_model.set_weights(global_model.get_weights())
        node = EdgeNode(
            factory_id=fid,
            model=local_model,
            mean_shift=dist["mean_shift"],
            noise_scale=dist["noise_scale"],
            confidence_threshold=e_cfg["confidence_threshold"],
            sampling_rate_normal=e_cfg["sampling_rate_normal"],
            input_dim=e_cfg["input_dim"],
            rng_seed=i * 42,
        )
        trainer = LocalTrainer(
            factory_id=fid,
            model=local_model,
            learning_rate=m_cfg["learning_rate"],
            local_epochs=e_cfg["local_epochs"],
            batch_size=e_cfg["local_batch_size"],
        )
        edge_nodes.append(node)
        local_trainers.append(trainer)

    pubsub = PubSubSimulator(max_queue_size=c_cfg["pubsub"]["max_queue_size"])
    feature_store = FeatureStore(feature_dim=m_cfg["input_dim"])
    drift_detector = DriftDetector(
        significance=c_cfg["drift_detector"]["ks_significance"],
        window_size=c_cfg["drift_detector"]["window_size"],
    )
    aggregator = FederatedAggregator(
        global_model=global_model,
        aggregation=g_cfg["federated_learning"]["aggregation"],
        min_clients=g_cfg["federated_learning"]["min_clients"],
    )
    registry = ModelRegistry(max_versions=g_cfg["model_registry"]["max_versions"])
    pipeline = MLOpsPipeline(
        drift_trigger_threshold=g_cfg["mlops_pipeline"]["drift_trigger_threshold"],
        performance_drop_threshold=g_cfg["mlops_pipeline"]["performance_drop_threshold"],
    )
    active_learner = ActiveLearner(
        model=global_model,
        strategy=c_cfg["active_learning"]["uncertainty_strategy"],
        budget=c_cfg["active_learning"]["budget_per_round"],
    )

    return {
        "global_model": global_model,
        "edge_nodes": edge_nodes,
        "local_trainers": local_trainers,
        "pubsub": pubsub,
        "feature_store": feature_store,
        "drift_detector": drift_detector,
        "aggregator": aggregator,
        "registry": registry,
        "pipeline": pipeline,
        "active_learner": active_learner,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot helpers – capture current component state for the UI
# ──────────────────────────────────────────────────────────────────────────────

def _snap(state: PipelineState, comps: Dict[str, Any]) -> None:
    """Persist component stats snapshots so the UI/agent can read them."""
    edge_nodes = comps["edge_nodes"]
    state.set_snapshot("edge_nodes", {n.factory_id: n.stats() for n in edge_nodes})
    state.set_snapshot("feature_store", comps["feature_store"].stats())
    state.set_snapshot("drift_detector", comps["drift_detector"].stats())
    state.set_snapshot("active_learner", comps["active_learner"].stats())
    state.set_snapshot("pubsub", comps["pubsub"].stats())
    state.set_snapshot("model_registry", {
        "versions": comps["registry"].list_versions(),
        "production_version": comps["registry"].get_production_version_id(),
    })
    state.set_snapshot("mlops_pipeline", {
        "runs": comps["pipeline"].list_runs(),
        "total_runs": comps["pipeline"].total_runs,
    })
    state.set_snapshot("aggregator", {
        "current_round": comps["aggregator"].current_round,
        "history": comps["aggregator"].get_history(),
    })


# ──────────────────────────────────────────────────────────────────────────────
# Round implementations (mirror main.py logic, emit events)
# ──────────────────────────────────────────────────────────────────────────────

def _run_bootstrap(state: PipelineState, comps: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    rd = 0
    state.set_round(rd)
    state.emit(rd, "system", "Pipeline", "round_start",
               "🏁 Round 0 – Bootstrap (Initial Data Collection + Baseline Model)")

    d_cfg = cfg["data"]
    feature_store = comps["feature_store"]
    drift_detector = comps["drift_detector"]
    aggregator = comps["aggregator"]
    local_trainers = comps["local_trainers"]
    edge_nodes = comps["edge_nodes"]
    registry = comps["registry"]
    pipeline = comps["pipeline"]
    active_learner = comps["active_learner"]

    all_X, all_y = [], []

    # Edge data generation
    for node, trainer in zip(edge_nodes, local_trainers):
        X, y = node.generate_batch(
            n_samples=d_cfg["samples_per_factory"],
            defect_rate=d_cfg["defect_rate_base"],
        )
        feature_store.ingest_batch(X, y, factory_id=node.factory_id, source="bootstrap")
        all_X.append(X)
        all_y.append(y)
        state.emit(rd, "edge", "EdgeNode", "data_generated",
                   f"{node.factory_id}: generated {len(X)} samples (defects: {int(y.sum())})",
                   {"factory_id": node.factory_id, "n_samples": len(X),
                    "defects": int(y.sum()), "normal": int((y == 0).sum())})

    # Drift reference
    X_ref = np.vstack(all_X)
    drift_detector.set_reference(X_ref)
    state.emit(rd, "cloud", "DriftDetector", "reference_set",
               f"Drift detector: reference set with {len(X_ref)} samples",
               {"n_samples": len(X_ref)})

    # FL baseline
    X_train, y_train = feature_store.get_training_data(labeled_only=True)
    state.emit(rd, "cloud", "FeatureStore", "training_data",
               f"Feature Store: retrieved {len(X_train)} samples for training",
               {"n_samples": len(X_train)})

    client_updates = []
    for node, trainer in zip(edge_nodes, local_trainers):
        X_fac, y_fac = feature_store.get_training_data(factory_ids=[node.factory_id])
        result = trainer.train(X_fac, y_fac, global_weights=aggregator.get_global_weights())
        client_updates.append(result)
        state.emit(rd, "edge", "LocalTrainer", "local_training",
                   f"{node.factory_id}: loss={result['train_loss']:.4f}  acc={result['train_acc']:.4f}",
                   {"factory_id": node.factory_id, "n_samples": result["n_samples"],
                    "train_loss": result["train_loss"], "train_acc": result["train_acc"]})

    agg_result = aggregator.aggregate(client_updates)
    state.emit(rd, "global", "FederatedAggregator", "aggregation",
               f"FedAvg round {agg_result['round']}: avg_loss={agg_result['avg_train_loss']:.4f}  avg_acc={agg_result['avg_train_acc']:.4f}",
               {"round": agg_result["round"], "avg_train_loss": agg_result["avg_train_loss"],
                "avg_train_acc": agg_result["avg_train_acc"], "n_clients": agg_result["n_clients"]})

    # Evaluate baseline
    X_all, y_all = feature_store.get_training_data()
    split = int(len(X_all) * 0.8)
    X_val, y_val = X_all[split:], y_all[split:]
    eval_result = local_trainers[0].evaluate(X_val, y_val)
    baseline_acc = eval_result["accuracy"]
    pipeline.set_baseline_accuracy(baseline_acc)
    state.emit(rd, "global", "Evaluator", "evaluation",
               f"Baseline model – accuracy={baseline_acc:.4f}  f1={eval_result['f1']:.4f}",
               {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, float)})

    # Register
    version_id = registry.register(
        model=aggregator.global_model, fl_round=0,
        metrics={"accuracy": baseline_acc, "f1": eval_result["f1"]},
        tags={"trigger": "bootstrap"},
        description="Initial global model – bootstrap round",
    )
    registry.promote_to_production(version_id, strategy="direct")
    state.emit(rd, "global", "ModelRegistry", "model_registered",
               f"Model {version_id} registered → promoted to PRODUCTION (direct)",
               {"version_id": version_id, "strategy": "direct", "accuracy": baseline_acc})

    # Distribute to edges
    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    active_learner.update_model(aggregator.global_model)

    state.emit(rd, "system", "Pipeline", "round_end",
               "Round 0 complete – baseline model deployed ✓")
    _snap(state, comps)


def _run_fl_round(state: PipelineState, comps: Dict[str, Any], cfg: Dict[str, Any], fl_round: int) -> None:
    rd = fl_round
    state.set_round(rd)
    state.emit(rd, "system", "Pipeline", "round_start",
               f"🔁 Round {rd} – Federated Learning (FedAvg across 3 factories)")

    e_cfg = cfg["edge"]
    d_cfg = cfg["data"]
    edge_nodes = comps["edge_nodes"]
    local_trainers = comps["local_trainers"]
    feature_store = comps["feature_store"]
    pubsub = comps["pubsub"]
    aggregator = comps["aggregator"]
    drift_detector = comps["drift_detector"]
    registry = comps["registry"]
    pipeline = comps["pipeline"]
    active_learner = comps["active_learner"]

    client_updates = []
    for node, trainer in zip(edge_nodes, local_trainers):
        X, y = node.generate_batch(
            n_samples=e_cfg["inference_latency_ms"],
            defect_rate=d_cfg["defect_rate_base"],
        )
        infer = node.run_inference(X)
        state.emit(rd, "edge", "EdgeNode", "inference",
                   f"{node.factory_id}: inferred {len(X)} samples  latency={infer['latency_ms']:.1f}ms",
                   {"factory_id": node.factory_id, "n_samples": len(X),
                    "latency_ms": infer["latency_ms"],
                    "defects_detected": int(infer["predictions"].sum())})

        upload = node.filter_for_upload(X, y, infer)
        pubsub.publish(
            topic="edge-data-upload", factory_id=node.factory_id,
            event_type="data_upload",
            payload={"n_samples": upload["n_uploaded"],
                     "reasons": upload["reasons"][:5]},
        )
        state.emit(rd, "edge", "EdgeNode", "upload",
                   f"{node.factory_id}: uploaded {upload['n_uploaded']}/{upload['n_total']} samples",
                   {"factory_id": node.factory_id, "n_uploaded": upload["n_uploaded"],
                    "n_total": upload["n_total"]})

        feature_store.ingest_batch(upload["X"], upload["y"],
                                   factory_id=node.factory_id, source="edge_upload")
        drift_detector.update(upload["X"])

        train_result = trainer.train(upload["X"], upload["y"],
                                     global_weights=aggregator.get_global_weights())
        client_updates.append(train_result)
        state.emit(rd, "edge", "LocalTrainer", "local_training",
                   f"{node.factory_id}: loss={train_result['train_loss']:.4f}  acc={train_result['train_acc']:.4f}",
                   {"factory_id": node.factory_id, "train_loss": train_result["train_loss"],
                    "train_acc": train_result["train_acc"], "n_samples": train_result["n_samples"]})

        pubsub.publish(topic="model-updates", factory_id=node.factory_id,
                       event_type="fl_weight_update",
                       payload={"n_samples": train_result["n_samples"], "fl_round": fl_round})

    # Aggregation
    agg_result = aggregator.aggregate(client_updates)
    if not agg_result["success"]:
        state.emit(rd, "global", "FederatedAggregator", "aggregation_failed",
                   f"Aggregation failed: {agg_result.get('reason')}")
        return

    state.emit(rd, "global", "FederatedAggregator", "aggregation",
               f"FedAvg round {agg_result['round']}: avg_acc={agg_result['avg_train_acc']:.4f}",
               {"round": agg_result["round"], "n_clients": agg_result["n_clients"],
                "total_samples": agg_result["total_samples"],
                "avg_train_loss": agg_result["avg_train_loss"],
                "avg_train_acc": agg_result["avg_train_acc"]})

    # Drift check
    drift = drift_detector.check_data_drift()
    psi = drift_detector.check_psi()
    state.emit(rd, "cloud", "DriftDetector", "drift_check",
               f"KS test: drift={drift['is_drift']}  PSI: drift={psi['is_drift']}",
               {"ks_drift": drift["is_drift"],
                "drift_fraction": drift.get("drift_fraction", 0),
                "min_p_value": drift.get("min_p_value", 1.0),
                "mean_psi": psi["mean_psi"], "psi_drift": psi["is_drift"]})

    # Evaluate
    X_all, y_all = feature_store.get_training_data()
    current_acc = agg_result["avg_train_acc"]
    if len(X_all) > 10:
        split = max(10, int(len(X_all) * 0.8))
        X_val, y_val = X_all[split:], y_all[split:]
        eval_result = local_trainers[0].evaluate(X_val, y_val)
        current_acc = eval_result["accuracy"]
        state.emit(rd, "global", "Evaluator", "evaluation",
                   f"Evaluation: acc={eval_result['accuracy']:.4f}  f1={eval_result['f1']:.4f}",
                   {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, float)})

    # Register
    version_id = registry.register(
        model=aggregator.global_model, fl_round=fl_round,
        metrics={"accuracy": current_acc, "avg_train_acc": agg_result["avg_train_acc"]},
        tags={"trigger": "fl_round"},
        description=f"Global model after FL round {fl_round}",
    )

    prod_vid = registry.get_production_version_id()
    prod_metrics = registry.get_version(prod_vid).metrics if prod_vid else {}
    if current_acc >= prod_metrics.get("accuracy", 0):
        registry.promote_to_production(version_id, strategy="canary")
        state.emit(rd, "global", "ModelRegistry", "model_promoted",
                   f"{version_id} promoted to PRODUCTION (canary) ✓",
                   {"version_id": version_id, "strategy": "canary", "accuracy": current_acc})
    else:
        state.emit(rd, "global", "ModelRegistry", "model_staged",
                   f"{version_id} kept in STAGING (accuracy did not improve)",
                   {"version_id": version_id, "accuracy": current_acc})

    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    active_learner.update_model(aggregator.global_model)

    state.emit(rd, "system", "Pipeline", "round_end", f"Round {rd} complete ✓")
    _snap(state, comps)


def _run_drift_round(state: PipelineState, comps: Dict[str, Any], cfg: Dict[str, Any], fl_round: int) -> None:
    rd = fl_round
    state.set_round(rd)
    state.emit(rd, "system", "Pipeline", "round_start",
               f"⚠️  Round {rd} – Data Drift Simulation + Active Learning")

    e_cfg = cfg["edge"]
    edge_nodes = comps["edge_nodes"]
    local_trainers = comps["local_trainers"]
    feature_store = comps["feature_store"]
    pubsub = comps["pubsub"]
    drift_detector = comps["drift_detector"]
    aggregator = comps["aggregator"]
    registry = comps["registry"]
    pipeline_obj = comps["pipeline"]
    active_learner = comps["active_learner"]
    rng = np.random.default_rng(999)

    # Drifted data
    for node in edge_nodes:
        X_drift, y_drift = node.generate_batch(n_samples=200, defect_rate=0.30)
        X_drift = X_drift * 2.0 + rng.standard_normal(X_drift.shape).astype(np.float32) * 0.5

        infer = node.run_inference(X_drift)
        upload = node.filter_for_upload(X_drift, y_drift, infer)
        drift_detector.update(X_drift)
        feature_store.ingest_batch(upload["X"], None, factory_id=node.factory_id,
                                   source="edge_upload_unlabeled")
        state.emit(rd, "edge", "EdgeNode", "drifted_upload",
                   f"{node.factory_id}: drifted data uploaded  n={upload['n_uploaded']} (unlabeled)",
                   {"factory_id": node.factory_id, "n_uploaded": upload["n_uploaded"]})

    # Drift detection
    drift = drift_detector.check_data_drift()
    psi = drift_detector.check_psi()
    state.emit(rd, "cloud", "DriftDetector", "drift_check",
               f"KS test: drift={drift['is_drift']}  fraction={drift.get('drift_fraction', 0):.2%}  "
               f"PSI: mean={psi['mean_psi']:.4f}  drift={psi['is_drift']}",
               {"ks_drift": drift["is_drift"],
                "drift_fraction": drift.get("drift_fraction", 0),
                "min_p_value": drift.get("min_p_value", 1.0),
                "mean_psi": psi["mean_psi"], "psi_drift": psi["is_drift"]})

    if drift["is_drift"] or psi["is_drift"]:
        state.emit(rd, "cloud", "DriftDetector", "drift_alert",
                   "🚨 DRIFT DETECTED → publishing pipeline-trigger event",
                   {"drift": drift["is_drift"], "psi": psi["is_drift"]})
        pubsub.publish(topic="alerts", factory_id="all", event_type="drift_alert",
                       payload={"drift_result": drift, "psi_result": psi})
        pubsub.publish(topic="pipeline-triggers", factory_id="global",
                       event_type="retrain_trigger",
                       payload={"reason": "data_drift", "fl_round": fl_round})

    # Active Learning
    X_unlabeled, entity_ids = feature_store.get_unlabeled_data()
    state.emit(rd, "cloud", "ActiveLearner", "unlabeled_pool",
               f"Feature Store: {len(X_unlabeled)} unlabeled samples available",
               {"n_unlabeled": len(X_unlabeled)})

    if len(X_unlabeled) > 0:
        X_selected, selected_ids, scores = active_learner.select_samples(X_unlabeled, entity_ids)
        state.emit(rd, "cloud", "ActiveLearner", "samples_selected",
                   f"Active Learner: selected {len(X_selected)} samples (strategy={active_learner.strategy})",
                   {"n_selected": len(X_selected), "strategy": active_learner.strategy,
                    "top_scores": scores[:5].round(4).tolist()})

        pseudo_labels = active_learner.oracle_label(X_selected, noise=0.05, rng=rng)
        labeled_count = feature_store.label_entities(selected_ids, pseudo_labels.tolist())
        state.emit(rd, "cloud", "ActiveLearner", "oracle_labeled",
                   f"Vertex AI Data Labeling: annotated {labeled_count} samples (defects: {int(pseudo_labels.sum())})",
                   {"labeled_count": labeled_count, "defects": int(pseudo_labels.sum())})

    # MLOps Pipeline
    trigger = pipeline_obj.should_trigger(drift_result=drift, force=drift["is_drift"])
    if trigger:
        state.emit(rd, "global", "MLOpsPipeline", "pipeline_triggered",
                   f"Pipeline triggered: {trigger.value}",
                   {"trigger": trigger.value})

        _state: Dict[str, Any] = {}

        def step_data_validation():
            X_all, y_all = feature_store.get_training_data(labeled_only=True)
            _state["X_all"], _state["y_all"] = X_all, y_all
            return {"n_labeled": len(X_all), "n_unlabeled": feature_store.stats()["unlabeled"]}

        def step_feature_engineering():
            stats = feature_store.compute_feature_statistics()
            return {"n_samples": stats.get("n_samples", 0), "feature_dim": len(stats.get("mean", []))}

        def step_federated_training():
            client_updates = []
            for node, trainer in zip(edge_nodes, local_trainers):
                X_fac, y_fac = feature_store.get_training_data(
                    factory_ids=[node.factory_id], labeled_only=True)
                if len(X_fac) == 0:
                    continue
                result = trainer.train(X_fac, y_fac, global_weights=aggregator.get_global_weights())
                client_updates.append(result)
            _state["client_updates"] = client_updates
            if len(client_updates) >= 2:
                agg = aggregator.aggregate(client_updates)
                _state["agg_result"] = agg
                return {"fl_round": agg["round"], "avg_acc": agg["avg_train_acc"],
                        "clients": agg["n_clients"]}
            return {"clients": len(client_updates), "skipped": True}

        def step_model_evaluation():
            X_all, y_all = _state["X_all"], _state["y_all"]
            if len(X_all) < 10:
                return {"skipped": True}
            split = max(10, int(len(X_all) * 0.8))
            X_val, y_val = X_all[split:], y_all[split:]
            eval_result = local_trainers[0].evaluate(X_val, y_val)
            _state["eval_result"] = eval_result
            return {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, float)}

        def step_model_registration():
            er = _state.get("eval_result", {})
            vid = registry.register(
                model=aggregator.global_model, fl_round=fl_round,
                metrics={"accuracy": er.get("accuracy", 0), "f1": er.get("f1", 0)},
                tags={"trigger": "drift_retrain"},
                description=f"Retrained model after drift detection (round {fl_round})",
            )
            _state["version_id"] = vid
            return {"version_id": vid}

        def step_deployment():
            vid = _state.get("version_id")
            if vid:
                registry.promote_to_production(vid, strategy="shadow")
                return {"version_id": vid, "strategy": "shadow"}
            return {"skipped": True}

        def step_monitoring_setup():
            return {"drift_threshold": pipeline_obj.drift_trigger_threshold,
                    "perf_threshold": pipeline_obj.performance_drop_threshold}

        run = pipeline_obj.run(
            trigger_reason=trigger, fl_round=fl_round,
            step_executors={
                "data_validation": step_data_validation,
                "feature_engineering": step_feature_engineering,
                "federated_training": step_federated_training,
                "model_evaluation": step_model_evaluation,
                "model_registration": step_model_registration,
                "deployment": step_deployment,
                "monitoring_setup": step_monitoring_setup,
            },
        )
        run_dict = run.to_dict()
        state.emit(rd, "global", "MLOpsPipeline", "pipeline_completed",
                   f"Pipeline {run.run_id}: {run.status.value}  duration={run_dict.get('duration_ms', 0):.0f}ms",
                   run_dict)

    # Update edge nodes
    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    active_learner.update_model(aggregator.global_model)

    state.emit(rd, "system", "Pipeline", "round_end", f"Round {rd} complete ✓")
    _snap(state, comps)


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(state: PipelineState) -> None:
    """
    Execute the full 4-round pipeline, emitting events to *state*.

    Designed to be called from a background thread so that the Streamlit
    UI can poll `state` while the pipeline is running.
    """
    try:
        state.set_status("running")
        cfg = _load_config()

        state.emit(-1, "system", "Pipeline", "config_loaded",
                   f"Config loaded – {cfg['edge']['num_factories']} factories, "
                   f"{cfg['global']['federated_learning']['num_rounds']} FL rounds",
                   {"factories": cfg["edge"]["num_factories"],
                    "fl_rounds": cfg["global"]["federated_learning"]["num_rounds"]})

        comps = _setup_components(cfg)
        state.emit(-1, "system", "Pipeline", "components_ready",
                   "All components initialised ✓")

        # Round 0
        _run_bootstrap(state, comps, cfg)

        # Round 1
        _run_fl_round(state, comps, cfg, fl_round=1)

        # Round 2 – drift + active learning
        _run_drift_round(state, comps, cfg, fl_round=2)

        # Round 3 – post-drift FL
        _run_fl_round(state, comps, cfg, fl_round=3)

        # Final snapshots
        _snap(state, comps)
        state.set_status("completed")
        state.emit(3, "system", "Pipeline", "pipeline_complete",
                   "✅ Pipeline complete – all 4 rounds executed successfully")

    except Exception as exc:
        state.set_status("error", error=str(exc))
        state.emit(state.current_round, "system", "Pipeline", "error",
                   f"❌ Pipeline error: {exc}", {"error": str(exc)})
        raise

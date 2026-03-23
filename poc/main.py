"""
main.py – End-to-End Manufacturing AI POC

Demonstrates the full architecture described in:
  federated_learning_edge_to_cloud_gcp_architecture.md

Architecture layers simulated:
  🏭 EDGE              → EdgeNode (inference, data filtering)
                         LocalTrainer (federated local training)
  🌍 REGIONAL CLOUD    → PubSubSimulator (event-driven ingestion)
                         FeatureStore (cross-factory features)
                         ActiveLearner (uncertainty-based labeling)
                         DriftDetector (KS-test + concept drift)
  🧠 GLOBAL CONTROL    → FederatedAggregator (FedAvg)
                         ModelRegistry (versioning + deployment strategies)
                         MLOpsPipeline (CI/CD/CT orchestration)

Flow:
  Round 0 : Bootstrap – generate initial labeled data, train first global model
  Round 1 : FL round – local training on all 3 factories, FedAvg aggregation
  Round 2 : Simulate data drift → pipeline triggers retraining
            Active Learning selects uncertain samples for annotation
  Round 3 : Retrain with newly labeled data, register new version, deploy
            with Canary strategy, push global model back to edge nodes
"""

import sys
import os
import time
import numpy as np
import torch
import yaml
from typing import Dict, Any

# Make sure poc/ sub-packages are importable regardless of working directory
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def subsection(title: str) -> None:
    print(f"\n  ┌─ {title}")


def log(msg: str, indent: int = 4) -> None:
    print(" " * indent + msg)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Component setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_components(cfg: Dict[str, Any]):
    """Instantiate all POC components from config."""
    m_cfg = cfg["model"]
    e_cfg = cfg["edge"]
    c_cfg = cfg["cloud"]
    g_cfg = cfg["global"]
    d_cfg = cfg["data"]

    # ── Global model (all edge nodes start with a copy)
    global_model = DefectDetector(
        input_dim=m_cfg["input_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_classes=m_cfg["num_classes"],
    )

    # ── Edge nodes (3 factories)
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
        local_model.set_weights(global_model.get_weights())  # start from global

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

    # ── Regional Cloud
    pubsub = PubSubSimulator(max_queue_size=c_cfg["pubsub"]["max_queue_size"])
    feature_store = FeatureStore(feature_dim=m_cfg["input_dim"])
    drift_detector = DriftDetector(
        significance=c_cfg["drift_detector"]["ks_significance"],
        window_size=c_cfg["drift_detector"]["window_size"],
    )

    # ── Global Control Plane
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
# ROUND 0 – Bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap(components: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    Generate initial data, populate Feature Store, train a baseline global model.
    """
    section("🏁 ROUND 0 – Bootstrap (Initial Data Collection + Baseline Model)")

    e_cfg = cfg["edge"]
    d_cfg = cfg["data"]
    feature_store: FeatureStore = components["feature_store"]
    drift_detector: DriftDetector = components["drift_detector"]
    aggregator: FederatedAggregator = components["aggregator"]
    local_trainers = components["local_trainers"]
    edge_nodes = components["edge_nodes"]
    registry: ModelRegistry = components["registry"]
    pipeline: MLOpsPipeline = components["pipeline"]
    active_learner: ActiveLearner = components["active_learner"]

    all_X, all_y = [], []

    subsection("🏭 Edge nodes generating initial labeled data")
    for node, trainer in zip(edge_nodes, local_trainers):
        X, y = node.generate_batch(
            n_samples=d_cfg["samples_per_factory"],
            defect_rate=d_cfg["defect_rate_base"],
        )
        # Ingest all as labeled (bootstrap phase – fully supervised)
        feature_store.ingest_batch(X, y, factory_id=node.factory_id, source="bootstrap")
        all_X.append(X)
        all_y.append(y)
        log(f"{node.factory_id}: generated {len(X)} samples  "
            f"(defects: {int(y.sum())}, normal: {int((y==0).sum())})")

    # Set drift reference distribution
    X_ref = np.vstack(all_X)
    drift_detector.set_reference(X_ref)
    log(f"\n  Drift detector: reference set with {len(X_ref)} samples")

    subsection("🌍 Regional Cloud – Initial Vertex AI Training")
    # Retrieve all labeled data from Feature Store
    X_train, y_train = feature_store.get_training_data(labeled_only=True)
    log(f"Feature Store: retrieved {len(X_train)} samples for training")

    # Run ONE FL round to get baseline global model
    client_updates = []
    for node, trainer in zip(edge_nodes, local_trainers):
        X_fac, y_fac = feature_store.get_training_data(factory_ids=[node.factory_id])
        result = trainer.train(X_fac, y_fac, global_weights=aggregator.get_global_weights())
        client_updates.append(result)
        log(f"  {node.factory_id}: trained on {result['n_samples']} samples  "
            f"loss={result['train_loss']:.4f}  acc={result['train_acc']:.4f}")

    agg_result = aggregator.aggregate(client_updates)
    log(f"\n  FedAvg round {agg_result['round']}: "
        f"avg_loss={agg_result['avg_train_loss']:.4f}  "
        f"avg_acc={agg_result['avg_train_acc']:.4f}")

    # Evaluate baseline model
    X_all, y_all = feature_store.get_training_data()
    split = int(len(X_all) * 0.8)
    X_val, y_val = X_all[split:], y_all[split:]
    eval_result = local_trainers[0].evaluate(X_val, y_val)
    baseline_acc = eval_result["accuracy"]
    pipeline.set_baseline_accuracy(baseline_acc)
    log(f"  Baseline model – accuracy={baseline_acc:.4f}  "
        f"f1={eval_result['f1']:.4f}  "
        f"precision={eval_result['precision']:.4f}  "
        f"recall={eval_result['recall']:.4f}")

    # Register in Model Registry
    version_id = registry.register(
        model=aggregator.global_model,
        fl_round=0,
        metrics={"accuracy": baseline_acc, "f1": eval_result["f1"]},
        tags={"trigger": "bootstrap"},
        description="Initial global model – bootstrap round",
    )
    registry.promote_to_production(version_id, strategy="direct")
    log(f"\n  Model Registry: registered {version_id} → promoted to PRODUCTION (direct)")

    # Distribute global model back to all edge nodes
    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    log("  Global model distributed to all 3 factory edge nodes ✓")

    # Update Active Learner's model reference
    active_learner.update_model(aggregator.global_model)


# ──────────────────────────────────────────────────────────────────────────────
# ROUND 1 – Federated Learning
# ──────────────────────────────────────────────────────────────────────────────

def federated_learning_round(
    components: Dict[str, Any],
    cfg: Dict[str, Any],
    fl_round: int,
) -> None:
    section(f"🔁 ROUND {fl_round} – Federated Learning (FedAvg across 3 factories)")

    e_cfg = cfg["edge"]
    d_cfg = cfg["data"]
    edge_nodes = components["edge_nodes"]
    local_trainers = components["local_trainers"]
    feature_store: FeatureStore = components["feature_store"]
    pubsub: PubSubSimulator = components["pubsub"]
    aggregator: FederatedAggregator = components["aggregator"]
    drift_detector: DriftDetector = components["drift_detector"]
    registry: ModelRegistry = components["registry"]
    pipeline: MLOpsPipeline = components["pipeline"]
    active_learner: ActiveLearner = components["active_learner"]

    subsection("🏭 Edge: inference + data filtering + Pub/Sub publish")
    client_updates = []
    total_uploaded = 0

    for node, trainer in zip(edge_nodes, local_trainers):
        # Generate new production data
        X, y = node.generate_batch(
            n_samples=e_cfg["inference_latency_ms"],  # ~80 samples per cycle
            defect_rate=d_cfg["defect_rate_base"],
        )

        # Inference
        infer = node.run_inference(X)
        log(f"{node.factory_id}: inferred {len(X)} samples  "
            f"latency={infer['latency_ms']:.1f}ms  "
            f"defects_detected={infer['predictions'].sum()}")

        # Filter for upload
        upload = node.filter_for_upload(X, y, infer)
        total_uploaded += upload["n_uploaded"]

        # Publish to Pub/Sub (simulates edge → regional cloud)
        pubsub.publish(
            topic="edge-data-upload",
            factory_id=node.factory_id,
            event_type="data_upload",
            payload={
                "n_samples": upload["n_uploaded"],
                "reasons": upload["reasons"][:5],  # truncate for log
            },
        )
        log(f"  → uploaded {upload['n_uploaded']}/{upload['n_total']} samples "
            f"(defects+uncertain+monitoring)")

        # Ingest into Feature Store (regional cloud)
        feature_store.ingest_batch(
            upload["X"], upload["y"],
            factory_id=node.factory_id,
            source="edge_upload",
        )

        # Drift monitoring: update window
        drift_detector.update(upload["X"])

        # Local federated training (NO raw data leaves factory)
        train_result = trainer.train(
            upload["X"], upload["y"],
            global_weights=aggregator.get_global_weights(),
        )
        client_updates.append(train_result)
        log(f"  local training: loss={train_result['train_loss']:.4f}  "
            f"acc={train_result['train_acc']:.4f}  "
            f"samples={train_result['n_samples']}")

        # Publish weight update to global plane
        pubsub.publish(
            topic="model-updates",
            factory_id=node.factory_id,
            event_type="fl_weight_update",
            payload={"n_samples": train_result["n_samples"], "fl_round": fl_round},
        )

    subsection("🧠 Global Control Plane: FedAvg aggregation")
    agg_result = aggregator.aggregate(client_updates)
    if not agg_result["success"]:
        log(f"⚠ Aggregation failed: {agg_result.get('reason')}")
        return

    log(f"FedAvg round {agg_result['round']}:")
    log(f"  clients={agg_result['n_clients']}  "
        f"total_samples={agg_result['total_samples']}")
    log(f"  avg_loss={agg_result['avg_train_loss']:.4f}  "
        f"avg_acc={agg_result['avg_train_acc']:.4f}")
    for c in agg_result["client_contributions"]:
        log(f"    {c['factory_id']}: weight={c['weight']:.3f}  "
            f"acc={c['train_acc']:.4f}  n={c['n_samples']}")

    subsection("🌍 Regional Cloud: drift check")
    drift = drift_detector.check_data_drift()
    psi = drift_detector.check_psi()
    log(f"KS test: is_drift={drift['is_drift']}  "
        f"drift_fraction={drift.get('drift_fraction', 0):.2%}  "
        f"min_p={drift.get('min_p_value', 1.0):.4f}")
    log(f"PSI:     mean_psi={psi['mean_psi']:.4f}  "
        f"is_drift={psi['is_drift']}")

    # Evaluate updated global model
    X_all, y_all = feature_store.get_training_data()
    if len(X_all) > 10:
        split = max(10, int(len(X_all) * 0.8))
        X_val, y_val = X_all[split:], y_all[split:]
        eval_result = local_trainers[0].evaluate(X_val, y_val)
        log(f"Evaluation: acc={eval_result['accuracy']:.4f}  "
            f"f1={eval_result['f1']:.4f}  "
            f"precision={eval_result['precision']:.4f}  "
            f"recall={eval_result['recall']:.4f}")
        current_acc = eval_result["accuracy"]
    else:
        current_acc = agg_result["avg_train_acc"]

    # Register new version
    version_id = registry.register(
        model=aggregator.global_model,
        fl_round=fl_round,
        metrics={"accuracy": current_acc, "avg_train_acc": agg_result["avg_train_acc"]},
        tags={"trigger": "fl_round"},
        description=f"Global model after FL round {fl_round}",
    )

    # Check if we should promote to production
    prod_vid = registry.get_production_version_id()
    prod_metrics = registry.get_version(prod_vid).metrics if prod_vid else {}
    if current_acc >= prod_metrics.get("accuracy", 0):
        registry.promote_to_production(version_id, strategy="canary")
        log(f"Model Registry: {version_id} promoted to PRODUCTION (canary deployment) ✓")
    else:
        log(f"Model Registry: {version_id} kept in STAGING (accuracy did not improve)")

    # Push updated global model back to edge nodes
    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    active_learner.update_model(aggregator.global_model)

    subsection("🔁 Pub/Sub stats")
    ps_stats = pubsub.stats()
    for topic, s in ps_stats.items():
        if s["total_published"] > 0:
            log(f"{topic}: published={s['total_published']}  queued={s['queue_size']}")


# ──────────────────────────────────────────────────────────────────────────────
# ROUND 2 – Simulate Data Drift + Active Learning
# ──────────────────────────────────────────────────────────────────────────────

def simulate_drift_and_active_learning(
    components: Dict[str, Any],
    cfg: Dict[str, Any],
    fl_round: int,
) -> None:
    section(f"⚠️  ROUND {fl_round} – Data Drift Simulation + Active Learning")

    e_cfg = cfg["edge"]
    d_cfg = cfg["data"]
    edge_nodes = components["edge_nodes"]
    local_trainers = components["local_trainers"]
    feature_store: FeatureStore = components["feature_store"]
    pubsub: PubSubSimulator = components["pubsub"]
    drift_detector: DriftDetector = components["drift_detector"]
    aggregator: FederatedAggregator = components["aggregator"]
    registry: ModelRegistry = components["registry"]
    pipeline: MLOpsPipeline = components["pipeline"]
    active_learner: ActiveLearner = components["active_learner"]
    rng = np.random.default_rng(999)

    subsection("🏭 Edge: generating DRIFTED data (new defect pattern)")
    # Simulate a new defect type causing distribution shift
    all_drifted_X = []
    drifted_uploads = []
    for node in edge_nodes:
        X_drift, y_drift = node.generate_batch(n_samples=200, defect_rate=0.30)
        # Inject drift: scale up features (simulates new sensor calibration)
        X_drift = X_drift * 2.0 + rng.standard_normal(X_drift.shape).astype(np.float32) * 0.5
        all_drifted_X.append(X_drift)

        infer = node.run_inference(X_drift)
        upload = node.filter_for_upload(X_drift, y_drift, infer)
        drifted_uploads.append((upload, node.factory_id))

        drift_detector.update(X_drift)
        feature_store.ingest_batch(
            upload["X"],
            None,  # labels unknown – needs annotation
            factory_id=node.factory_id,
            source="edge_upload_unlabeled",
        )
        log(f"{node.factory_id}: drifted data uploaded  "
            f"n={upload['n_uploaded']}  (unlabeled – needs annotation)")

    subsection("🌍 Regional Cloud: drift detection")
    drift = drift_detector.check_data_drift()
    psi = drift_detector.check_psi()
    log(f"KS test: is_drift={drift['is_drift']}  "
        f"drift_fraction={drift.get('drift_fraction', 0):.2%}  "
        f"min_p={drift.get('min_p_value', 1.0):.6f}")
    log(f"PSI:     mean_psi={psi['mean_psi']:.4f}  is_drift={psi['is_drift']}")

    if drift["is_drift"] or psi["is_drift"]:
        log("🚨 DRIFT DETECTED → publishing pipeline-trigger event")
        pubsub.publish(
            topic="alerts",
            factory_id="all",
            event_type="drift_alert",
            payload={"drift_result": drift, "psi_result": psi},
        )
        pubsub.publish(
            topic="pipeline-triggers",
            factory_id="global",
            event_type="retrain_trigger",
            payload={"reason": "data_drift", "fl_round": fl_round},
        )

    subsection("🌍 Active Learning: select uncertain samples for annotation")
    X_unlabeled, entity_ids = feature_store.get_unlabeled_data()
    log(f"Feature Store: {len(X_unlabeled)} unlabeled samples available")

    if len(X_unlabeled) > 0:
        X_selected, selected_ids, scores = active_learner.select_samples(X_unlabeled, entity_ids)
        log(f"Active Learner: selected {len(X_selected)} samples  "
            f"(strategy={active_learner.strategy})")
        log(f"  top uncertainty scores: {scores[:5].round(4).tolist()}")

        # Simulate human annotation (oracle labeling)
        pseudo_labels = active_learner.oracle_label(X_selected, noise=0.05, rng=rng)
        labeled_count = feature_store.label_entities(selected_ids, pseudo_labels.tolist())
        log(f"  Vertex AI Data Labeling: annotated {labeled_count} samples  "
            f"(defects: {int(pseudo_labels.sum())})")
    else:
        log("  No unlabeled samples available yet")

    subsection("🧠 MLOps Pipeline: auto-triggered by drift")
    trigger = pipeline.should_trigger(drift_result=drift, force=drift["is_drift"])
    if trigger:
        log(f"Pipeline trigger: {trigger.value}")

        # State captured for step executors
        _state: Dict[str, Any] = {}

        def step_data_validation() -> Dict[str, Any]:
            X_all, y_all = feature_store.get_training_data(labeled_only=True)
            _state["X_all"], _state["y_all"] = X_all, y_all
            return {"n_labeled": len(X_all), "n_unlabeled": feature_store.stats()["unlabeled"]}

        def step_feature_engineering() -> Dict[str, Any]:
            stats = feature_store.compute_feature_statistics()
            return {"n_samples": stats.get("n_samples", 0), "feature_dim": len(stats.get("mean", []))}

        def step_federated_training() -> Dict[str, Any]:
            X_all = _state["X_all"]
            log(f"Feature Store: {len(X_all)} labeled samples available for retraining")
            client_updates = []
            for node, trainer in zip(edge_nodes, local_trainers):
                X_fac, y_fac = feature_store.get_training_data(
                    factory_ids=[node.factory_id], labeled_only=True
                )
                if len(X_fac) == 0:
                    continue
                result = trainer.train(X_fac, y_fac, global_weights=aggregator.get_global_weights())
                client_updates.append(result)
                log(f"  {node.factory_id}: retrain  loss={result['train_loss']:.4f}  "
                    f"acc={result['train_acc']:.4f}  n={result['n_samples']}")
            _state["client_updates"] = client_updates
            if len(client_updates) >= 2:
                agg = aggregator.aggregate(client_updates)
                _state["agg_result"] = agg
                return {"fl_round": agg["round"], "avg_acc": agg["avg_train_acc"], "clients": agg["n_clients"]}
            return {"clients": len(client_updates), "skipped": True}

        def step_model_evaluation() -> Dict[str, Any]:
            X_all, y_all = _state["X_all"], _state["y_all"]
            if len(X_all) < 10:
                return {"skipped": True}
            split = max(10, int(len(X_all) * 0.8))
            X_val, y_val = X_all[split:], y_all[split:]
            eval_result = local_trainers[0].evaluate(X_val, y_val)
            _state["eval_result"] = eval_result
            log(f"  Evaluation: acc={eval_result['accuracy']:.4f}  f1={eval_result['f1']:.4f}")
            return {k: round(v, 4) for k, v in eval_result.items() if isinstance(v, float)}

        def step_model_registration() -> Dict[str, Any]:
            er = _state.get("eval_result", {})
            version_id = registry.register(
                model=aggregator.global_model,
                fl_round=fl_round,
                metrics={"accuracy": er.get("accuracy", 0), "f1": er.get("f1", 0)},
                tags={"trigger": "drift_retrain"},
                description=f"Retrained model after drift detection (round {fl_round})",
            )
            _state["version_id"] = version_id
            return {"version_id": version_id}

        def step_deployment() -> Dict[str, Any]:
            vid = _state.get("version_id")
            if vid:
                registry.promote_to_production(vid, strategy="shadow")
                log(f"\n  Model Registry: {vid} promoted to PRODUCTION (shadow mode) ✓")
                log("  → Running in shadow mode: no live impact while we validate")
                return {"version_id": vid, "strategy": "shadow"}
            return {"skipped": True}

        def step_monitoring_setup() -> Dict[str, Any]:
            return {"drift_threshold": pipeline.drift_trigger_threshold,
                    "perf_threshold": pipeline.performance_drop_threshold}

        run = pipeline.run(
            trigger_reason=trigger,
            fl_round=fl_round,
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
        log(f"\n  Pipeline {run.run_id}: {run.status.value}  "
            f"duration={((run.end_time or 0) - run.start_time)*1000:.0f}ms")

    # Update edge nodes
    global_weights = aggregator.get_global_weights()
    for node in edge_nodes:
        node.update_model(global_weights)
    active_learner.update_model(aggregator.global_model)


# ──────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(components: Dict[str, Any]) -> None:
    section("📊 POC SUMMARY")

    edge_nodes = components["edge_nodes"]
    feature_store: FeatureStore = components["feature_store"]
    drift_detector: DriftDetector = components["drift_detector"]
    registry: ModelRegistry = components["registry"]
    pipeline: MLOpsPipeline = components["pipeline"]
    active_learner: ActiveLearner = components["active_learner"]
    pubsub: PubSubSimulator = components["pubsub"]

    subsection("🏭 Edge nodes")
    for node in edge_nodes:
        s = node.stats()
        log(f"{s['factory_id']}: inferred={s['total_inferred']}  "
            f"defects={s['defects_detected']}  "
            f"defect_rate={s['defect_rate']:.2%}  "
            f"upload_rate={s['upload_rate']:.2%}")

    subsection("🌍 Feature Store")
    fs_stats = feature_store.stats()
    log(f"total_entities={fs_stats['total_entities']}  "
        f"labeled={fs_stats['labeled']}  "
        f"unlabeled={fs_stats['unlabeled']}  "
        f"label_rate={fs_stats['label_rate']:.2%}")
    log(f"by_factory: {fs_stats['by_factory']}")

    subsection("🌍 Drift Detector")
    dd_stats = drift_detector.stats()
    log(f"checks={dd_stats['checks_performed']}  "
        f"drift_events={dd_stats['drift_events_detected']}  "
        f"reference_size={dd_stats['reference_size']}  "
        f"window_size={dd_stats['current_window_size']}")

    subsection("🌍 Active Learner")
    al_stats = active_learner.stats()
    log(f"strategy={al_stats['strategy']}  "
        f"rounds={al_stats['rounds_completed']}  "
        f"total_selected={al_stats['total_samples_selected']}")

    subsection("🧠 Model Registry – all versions")
    for v in registry.list_versions():
        stage_icon = "✅" if v["stage"] == "production" else "📦"
        log(f"{stage_icon} {v['version_id']:5s}  stage={v['stage']:10s}  "
            f"fl_round={v['fl_round']}  "
            f"acc={v['metrics'].get('accuracy', 0):.4f}  "
            f"strategy={v['deployment_strategy']}  "
            f"→ {v['description']}")

    subsection("🧠 MLOps Pipeline runs")
    for run in pipeline.list_runs():
        log(f"{run['run_id']}: trigger={run['trigger_reason']:20s}  "
            f"status={run['status']:10s}  "
            f"duration={run.get('duration_ms', 0):.0f}ms")
        for step in run["steps"]:
            icon = "✓" if step["status"] == "succeeded" else "✗"
            out = step.get("output", {})
            out_str = "  " + str({k: v for k, v in out.items() if k != "skipped"}) if out else ""
            log(f"    {icon} {step['name']:25s}  {step['duration_ms']:.0f}ms{out_str}")

    subsection("🔁 Pub/Sub totals")
    for topic, s in pubsub.stats().items():
        if s["total_published"] > 0:
            log(f"{topic}: total_published={s['total_published']}")

    section("✅ POC COMPLETE")
    prod_vid = registry.get_production_version_id()
    log(f"Production model: {prod_vid}")
    log("")
    log("Architecture layers demonstrated:")
    log("  🏭 Edge:    inference (<100ms), data filtering, local FL training")
    log("  🌍 Cloud:   Pub/Sub ingestion, Feature Store, Active Learning, Drift Detection")
    log("  🧠 Global:  FedAvg aggregation, Model Registry, MLOps Pipeline (CI/CD/CT)")
    log("  🔁 Deploy:  Canary → Shadow → Production lifecycle")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(config_path)

    print("\n" + "█" * 72)
    print("  🚀 Manufacturing AI POC – Federated Learning + Edge + MLOps on GCP")
    print("█" * 72)
    print(f"\n  Factories : {cfg['edge']['num_factories']}")
    print(f"  FL rounds : {cfg['global']['federated_learning']['num_rounds']}")
    print(f"  Aggregation: {cfg['global']['federated_learning']['aggregation']}")
    print(f"  AL strategy: {cfg['cloud']['active_learning']['uncertainty_strategy']}")

    # Initialise all components
    components = setup_components(cfg)

    # ── Round 0: Bootstrap
    bootstrap(components, cfg)

    # ── Round 1: Standard FL round
    federated_learning_round(components, cfg, fl_round=1)

    # ── Round 2: Simulate drift → active learning → retrain
    simulate_drift_and_active_learning(components, cfg, fl_round=2)

    # ── Round 3: Another standard FL round (post-drift)
    federated_learning_round(components, cfg, fl_round=3)

    # ── Summary
    print_summary(components)


if __name__ == "__main__":
    main()

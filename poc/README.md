# 🚀 Manufacturing AI POC

End-to-end Proof of Concept for the three-layer manufacturing AI architecture described in
[`federated_learning_edge_to_cloud_gcp_architecture.md`](../federated_learning_edge_to_cloud_gcp_architecture.md).

Runs entirely locally – **no GCP credentials required**.  All GCP services (Pub/Sub, Vertex AI Feature Store, Vertex AI Pipelines, etc.) are simulated with Python classes.

---

## Architecture Layers

```
🏭 EDGE (3 simulated factories)
   EdgeNode          – sensor simulation, <100ms inference, defect filtering
   LocalTrainer      – federated local training (no raw data leaves factory)

🌍 REGIONAL CLOUD
   PubSubSimulator   – event-driven data ingestion (Cloud Pub/Sub)
   FeatureStore      – cross-factory feature reuse (Vertex AI Feature Store)
   ActiveLearner     – entropy-based uncertainty sampling (Vertex AI Data Labeling)
   DriftDetector     – KS-test + PSI data drift & concept drift detection

🧠 GLOBAL CONTROL PLANE
   FederatedAggregator – FedAvg / weighted-FedAvg aggregation
   ModelRegistry       – versioning, staging/production lifecycle (Vertex AI Model Registry)
   MLOpsPipeline       – 7-step CI/CD/CT orchestration (Vertex AI Pipelines)
```

---

## Demo Flow

| Round | What happens |
|-------|-------------|
| **0 – Bootstrap** | 3 factories generate 500 samples each → Feature Store ingestion → initial FedAvg round → baseline model registered & deployed (direct) |
| **1 – FL Round** | Each edge node infers, filters uploads (defects + uncertain + monitoring sample), trains locally, sends weights → FedAvg → new version evaluated, canary promotion |
| **2 – Drift + Active Learning** | Injected data distribution shift (×2 scale) → KS test & PSI detect drift → Pub/Sub alert → Active Learner selects top-20 uncertain samples for annotation → MLOps pipeline auto-triggered (7 steps: validate → feature eng → FL train → eval → register → deploy → monitor) → shadow-mode promotion |
| **3 – Post-Drift FL** | Standard FL round with newly labeled data |

---

## Prerequisites

Python 3.10+

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `scikit-learn`, `scipy`, `pyyaml`, `pandas`, `matplotlib`, `streamlit`

---

## Run

### CLI mode

```bash
cd poc/
python main.py
```

Expected runtime: ~5–15 seconds (CPU only).

### Web UI (Streamlit)

```bash
cd poc/
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and provides:

- **▶️ Start Pipeline** – run the full 4-round pipeline with live event streaming
- **📊 Component Detail** – real-time metrics for every layer (Edge, Cloud, Global)
- **🤖 Agent Query** – ask questions in natural language, e.g.:
  - *"What is the pipeline status?"*
  - *"Which model is in production?"*
  - *"Has drift been detected?"*
  - *"Show me edge node statistics"*
  - *"What happened in round 2?"*

---

## Configuration

Edit `config.yaml` to tune:

| Section | Key parameters |
|---------|---------------|
| `edge` | `num_factories`, `confidence_threshold`, `sampling_rate_normal` |
| `cloud.active_learning` | `budget_per_round`, `uncertainty_strategy` (entropy / margin / least_confident) |
| `cloud.drift_detector` | `ks_significance`, `window_size` |
| `global.federated_learning` | `aggregation` (fedavg / weighted_fedavg), `num_rounds` |
| `model` | `hidden_dim`, `learning_rate` |

---

## File Structure

```
poc/
├── main.py                        # End-to-end orchestration (CLI)
├── app.py                         # Streamlit real-time dashboard + agent chat
├── pipeline_state.py              # Thread-safe event/state store for the UI
├── pipeline_runner.py             # Pipeline execution wrapper (emits events)
├── agent.py                       # Natural-language query agent
├── config.yaml                    # All tuneable parameters
├── requirements.txt
├── models/
│   └── defect_detector.py         # MLP defect detection model (simulates TensorRT INT8 edge model)
├── edge/
│   ├── edge_node.py               # EdgeNode: inference + data filtering
│   └── local_trainer.py           # LocalTrainer: federated local training
├── cloud/
│   ├── pubsub_simulator.py        # PubSubSimulator: event-driven ingestion
│   ├── feature_store.py           # FeatureStore: cross-factory features
│   ├── active_learning.py         # ActiveLearner: uncertainty sampling
│   └── drift_detector.py          # DriftDetector: KS test + PSI + concept drift
└── global_plane/
    ├── federated_aggregator.py    # FederatedAggregator: FedAvg
    ├── model_registry.py          # ModelRegistry: versioning + deployment strategies
    └── mlops_pipeline.py          # MLOpsPipeline: 7-step CI/CD/CT orchestration
```

---

## Key Design Concepts Demonstrated

### Federated Learning (EU Compliance)
Raw data **never leaves** each factory.  Only model weight updates are sent to the global aggregator.  The `weighted_fedavg` aggregation mode gives larger factories more influence proportional to their sample count.

### Active Learning (Cost Reduction)
Instead of labeling all 260+ unlabeled samples after a drift event, `ActiveLearner` selects only the 20 most uncertain ones (via Shannon entropy).  This maps to **Vertex AI Data Labeling** in production.

### Drift Detection (Governance)
- **Data drift** – KS test per feature + Population Stability Index (PSI ≥ 0.2 = significant)
- **Concept drift** – accuracy monitoring against baseline
- On detection: Pub/Sub alert → `pipeline-triggers` topic → MLOps pipeline auto-starts

### MLOps Pipeline (CI/CD/CT)
7 steps executed by `MLOpsPipeline.run()`:
1. `data_validation` – check Feature Store completeness
2. `feature_engineering` – compute feature statistics
3. `federated_training` – FL round across all factories
4. `model_evaluation` – accuracy / F1 / precision / recall on held-out set
5. `model_registration` – register new version in Model Registry
6. `deployment` – promote with chosen strategy (canary / shadow / ab_test / direct)
7. `monitoring_setup` – configure drift & performance thresholds

### Deployment Strategies
| Strategy | Use case |
|----------|----------|
| `direct` | Bootstrap / emergency hotfix |
| `canary` | Gradual rollout to a subset of factories |
| `shadow` | Parallel validation with no production impact |
| `ab_test` | Split traffic between old and new model |

---

## Production Mapping (GCP Services)

| POC Component | GCP / Vertex AI Service |
|---------------|------------------------|
| `EdgeNode` inference | Vertex AI Edge Manager + TensorRT INT8 |
| `PubSubSimulator` | Cloud Pub/Sub |
| `FeatureStore` | Vertex AI Feature Store |
| `ActiveLearner` | Vertex AI Data Labeling |
| `DriftDetector` | Vertex AI Model Monitoring |
| `LocalTrainer` | On-device training (Coral / NVIDIA Jetson) |
| `FederatedAggregator` | Custom FL server on Cloud Run / GKE |
| `ModelRegistry` | Vertex AI Model Registry + Artifact Registry |
| `MLOpsPipeline` | Vertex AI Pipelines (Kubeflow) |
| Data storage | Cloud Storage (images) + BigQuery (structured) |
| Security | VPC Service Controls + IAM + CMEK |

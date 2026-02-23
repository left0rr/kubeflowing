# kubeflowing — GPON Router Predictive Maintenance (Lightweight Edge MLOps)

> **Final-year internship project.**  
> An end-to-end MLOps system that predicts GPON router hardware failures **7 days in advance** using XGBoost, tracks experiments with MLflow, monitors data drift with Evidently AI, serves real-time predictions via FastAPI, and calls a local Ollama LLM to generate plain-text diagnostic tickets when a failure is likely.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Component Interactions](#3-component-interactions)
4. [Resource Budget (16 GB RAM VM)](#4-resource-budget-16-gb-ram-vm)
5. [Prerequisites](#5-prerequisites)
6. [Quick Start — Docker Compose](#6-quick-start--docker-compose)
7. [Running the FastAPI Server Locally](#7-running-the-fastapi-server-locally)
8. [Training the Model (Kubeflow Pipeline)](#8-training-the-model-kubeflow-pipeline)
9. [Calling the Prediction API](#9-calling-the-prediction-api)
10. [Monitoring with Grafana](#10-monitoring-with-grafana)
11. [Environment Variables](#11-environment-variables)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Edge VM  (16 GB RAM)                         │
│                                                                     │
│  ┌──────────────┐   compile   ┌─────────────────────────────────┐  │
│  │  kfp v2      │ ──────────► │  Kubeflow Pipeline              │  │
│  │  train_      │             │  generate_data → train_and_log  │  │
│  │  pipeline.py │             │              → drift_report     │  │
│  └──────────────┘             └──────────┬──────────────────────┘  │
│                                          │ XGBoost model.xgb        │
│                                          ▼                          │
│  ┌──────────────┐   log        ┌─────────────────┐                 │
│  │  PostgreSQL  │ ◄──────────  │    MLflow        │                 │
│  │  (backend)   │              │  Tracking Server │                 │
│  └──────────────┘              └────────┬────────┘                 │
│  ┌──────────────┐   artifacts   │        │                          │
│  │  MinIO (S3)  │ ◄─────────────┘        │                          │
│  └──────────────┘                        │                          │
│                                          │ model artifact           │
│  ┌──────────────────────────────────────▼──────────────────────┐   │
│  │              FastAPI  /predict  (serving/app.py)             │   │
│  │  TelemetryInput ──► XGBoost predict_proba ──► prob > 0.7?   │   │
│  │                                               │              │   │
│  │                                        YES    ▼              │   │
│  │                              ┌───────────────────────────┐   │   │
│  │                              │  Ollama LLM (localhost)   │   │   │
│  │                              │  → Diagnostic ticket text │   │   │
│  │                              └───────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────┐  scrape  ┌────────────┐  visualise ┌──────────┐  │
│  │  Prometheus  │ ◄─────── │  /metrics  │            │ Grafana  │  │
│  └──────────────┘          └────────────┘            └──────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```
kubeflowing/
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Full infrastructure stack
├── pipeline/
│   └── train_pipeline.py         # kfp v2 pipeline definition
├── serving/
│   ├── app.py                    # FastAPI prediction service
│   └── Dockerfile                # Container image for serving
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources/      # Auto-provision Prometheus datasource
│       │   └── dashboards/       # Auto-provision dashboard loader
│       └── dashboards/
│           └── overview.json     # Pre-built Grafana dashboard
└── README.md
```

---

## 3. Component Interactions

| Component | Role | Talks to |
|-----------|------|----------|
| **Kubeflow Pipeline** (`pipeline/train_pipeline.py`) | Orchestrates data generation → XGBoost training → Evidently drift report | MLflow (logging), local filesystem (model artifact) |
| **MLflow** | Experiment tracking, metric storage, model registry | PostgreSQL (metadata), MinIO (artifacts) |
| **PostgreSQL** | Relational backend for MLflow run metadata | — |
| **MinIO** | S3-compatible object store for model artifacts and plots | — |
| **FastAPI** (`serving/app.py`) | Real-time inference endpoint (`POST /predict`) | XGBoost model file, Ollama API |
| **Ollama** | Local LLM (e.g. Mistral) for diagnostic ticket generation | — (runs on host) |
| **Prometheus** | Time-series metrics from FastAPI `/metrics` | FastAPI, MLflow |
| **Grafana** | Dashboards for request rate, failure alerts, model drift | Prometheus |

---

## 4. Resource Budget (16 GB RAM VM)

| Service | Typical RSS |
|---------|-------------|
| PostgreSQL (tuned) | ~128 MB |
| MinIO | ~150 MB |
| MLflow server | ~300 MB |
| FastAPI (uvicorn) | ~200 MB |
| Prometheus | ~150 MB |
| Grafana | ~150 MB |
| **Ollama + Mistral 7B Q4** | ~5,500 MB |
| OS + overhead | ~1,500 MB |
| **Total** | **~8 GB** (comfortable on 16 GB) |

---

## 5. Prerequisites

| Tool | Version |
|------|---------|
| Docker + Docker Compose v2 | ≥ 24 |
| Python | 3.11+ |
| Ollama | latest |
| (Optional) Kubeflow Pipelines | 2.x |

Install Ollama and pull a model:

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a lightweight model
ollama pull mistral
```

---

## 6. Quick Start — Docker Compose

```bash
# 1. Clone the repository
git clone https://github.com/left0rr/kubeflowing.git
cd kubeflowing

# 2. Start the full stack
docker compose up -d

# 3. Wait ~60 s for all health checks to pass, then verify
docker compose ps
```

**Service URLs after startup:**

| Service | URL | Credentials |
|---------|-----|-------------|
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| FastAPI docs | http://localhost:8000/docs | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

To tear down and remove volumes:

```bash
docker compose down -v
```

---

## 7. Running the FastAPI Server Locally

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Point to a trained model
export MODEL_PATH=/path/to/model.xgb

# Start the server
uvicorn serving.app:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation: http://localhost:8000/docs

---

## 8. Training the Model (Kubeflow Pipeline)

### Compile the pipeline

```bash
pip install -r requirements.txt
python pipeline/train_pipeline.py
# Outputs: pipeline/pipeline.yaml
```

### Submit to a KFP instance

```bash
pip install kfp
python - <<'EOF'
import kfp
client = kfp.Client(host="http://<your-kfp-host>:8080")
client.create_run_from_pipeline_package(
    "pipeline/pipeline.yaml",
    arguments={
        "mlflow_tracking_uri": "http://mlflow:5000",
        "mlflow_experiment": "gpon-predictive-maintenance",
    },
)
EOF
```

### Train locally (without KFP)

```bash
python - <<'EOF'
import pandas as pd, random, numpy as np
from xgboost import XGBClassifier
import mlflow, mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

random.seed(42)
n = 2000
df = pd.DataFrame({
    "rx_power_dbm":  [random.uniform(-30, -5)  for _ in range(n)],
    "temperature_c": [random.uniform(30, 80)   for _ in range(n)],
    "latency_ms":    [random.uniform(1, 50)    for _ in range(n)],
    "tx_power_dbm":  [random.uniform(-5, 3)    for _ in range(n)],
    "ber":           [random.uniform(0, 1e-3)  for _ in range(n)],
    "uptime_hours":  [random.uniform(0, 8760)  for _ in range(n)],
    "failure_7d":    [1 if random.random()<0.15 else 0 for _ in range(n)],
})
features = ["rx_power_dbm","temperature_c","latency_ms","tx_power_dbm","ber","uptime_hours"]
X, y = df[features], df["failure_7d"]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("gpon-predictive-maintenance")
with mlflow.start_run():
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          eval_metric="logloss", random_state=42)
    model.fit(X_tr, y_tr)
    preds  = model.predict(X_val)
    probs  = model.predict_proba(X_val)[:,1]
    mlflow.log_metric("accuracy", accuracy_score(y_val, preds))
    mlflow.log_metric("roc_auc",  roc_auc_score(y_val, probs))
    mlflow.xgboost.log_model(model, "model")
    model.save_model("model.xgb")
    print("Model saved to model.xgb")
EOF
export MODEL_PATH=model.xgb
```

---

## 9. Calling the Prediction API

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rx_power_dbm": -28.5,
    "temperature_c": 72.0,
    "latency_ms": 42.3,
    "tx_power_dbm": -1.2,
    "ber": 0.0008,
    "uptime_hours": 7200
  }' | python3 -m json.tool
```

**Example response (high-risk device):**

```json
{
  "failure_probability": 0.8731,
  "failure_predicted": true,
  "diagnostic_report": "DIAGNOSTIC TICKET #AUTO-2024\nRoot cause: Degraded optical receiver (RX power −28.5 dBm well below −27 dBm threshold) combined with elevated temperature (72 °C).\nImmediate action: Schedule emergency visit; inspect SFP module and fiber connector for contamination or physical damage.\nPreventive steps: (1) Clean optical connectors quarterly. (2) Improve rack ventilation—install additional fans if ambient > 35 °C. (3) Enable proactive OLT alarm for RX < −26 dBm."
}
```

When `failure_predicted` is `false` the `diagnostic_report` field is `null`.

---

## 10. Monitoring with Grafana

1. Open Grafana at http://localhost:3000 (admin / admin).
2. The **GPON Predictive Maintenance Overview** dashboard is pre-provisioned.
3. Key panels:
   - **Prediction Requests / sec** – throughput from the `/predict` endpoint.
   - **High-Risk Predictions** – count of devices flagged for imminent failure.

To add custom metrics, instrument `serving/app.py` with `prometheus-fastapi-instrumentator` and restart the `serving` container.

---

## 11. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model.xgb` | Path to the trained XGBoost model file |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `mistral` | Ollama model name to use for diagnostics |
| `FAILURE_THRESHOLD` | `0.7` | Probability threshold above which a failure is reported |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow tracking server URI |

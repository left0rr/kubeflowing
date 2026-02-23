"""
Kubeflow Pipeline (kfp v2 SDK) for GPON Router Predictive Maintenance.

This pipeline:
  1. Generates/loads simulated telemetry data.
  2. Trains an XGBoost classifier to predict hardware failures 7 days ahead.
  3. Logs metrics and the model artifact to MLflow.
  4. Generates an Evidently AI data-drift report comparing training vs. reference data.
"""

import os

import kfp
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output

# ---------------------------------------------------------------------------
# Pipeline component: data generation
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.0.0", "scikit-learn>=1.4.0"],
)
def generate_data(
    train_dataset: Output[Dataset],
    reference_dataset: Output[Dataset],
):
    """Simulate GPON telemetry data and split into train / reference sets."""
    import random
    import pandas as pd

    random.seed(42)
    n = 2000

    data = pd.DataFrame(
        {
            "rx_power_dbm": [random.uniform(-30, -5) for _ in range(n)],
            "temperature_c": [random.uniform(30, 80) for _ in range(n)],
            "latency_ms": [random.uniform(1, 50) for _ in range(n)],
            "tx_power_dbm": [random.uniform(-5, 3) for _ in range(n)],
            "ber": [random.uniform(0, 1e-3) for _ in range(n)],
            "uptime_hours": [random.uniform(0, 8760) for _ in range(n)],
            # label: 1 = failure within 7 days, 0 = no failure
            "failure_7d": [1 if random.random() < 0.15 else 0 for _ in range(n)],
        }
    )

    split = int(n * 0.8)
    data.iloc[:split].to_csv(train_dataset.path, index=False)
    data.iloc[split:].to_csv(reference_dataset.path, index=False)


# ---------------------------------------------------------------------------
# Pipeline component: model training + MLflow logging
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "xgboost>=2.0.0",
        "mlflow>=2.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "boto3>=1.34.0",
        "psycopg2-binary>=2.9.9",
    ],
)
def train_and_log(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment: str = "gpon-predictive-maintenance",
):
    """Train XGBoost, evaluate, and log everything to MLflow."""
    import mlflow
    import mlflow.xgboost
    import pandas as pd
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    df = pd.read_csv(train_dataset.path)
    feature_cols = ["rx_power_dbm", "temperature_c", "latency_ms", "tx_power_dbm", "ber", "uptime_hours"]
    X = df[feature_cols]
    y = df["failure_7d"]

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }
        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        accuracy = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("roc_auc", auc)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"accuracy={accuracy:.4f}  roc_auc={auc:.4f}  run_id={run.info.run_id}")

    # Persist model locally so downstream components can consume it
    model.save_model(model_artifact.path)


# ---------------------------------------------------------------------------
# Pipeline component: Evidently data-drift report
# ---------------------------------------------------------------------------


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["evidently>=0.4.16", "pandas>=2.0.0"],
)
def drift_report(
    train_dataset: Input[Dataset],
    reference_dataset: Input[Dataset],
    report_path: Output[Dataset],
):
    """Generate an Evidently AI data-drift HTML report."""
    import pandas as pd
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    feature_cols = ["rx_power_dbm", "temperature_c", "latency_ms", "tx_power_dbm", "ber", "uptime_hours"]

    reference = pd.read_csv(reference_dataset.path)[feature_cols]
    current = pd.read_csv(train_dataset.path)[feature_cols]

    column_mapping = ColumnMapping()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    report.save_html(report_path.path)
    print(f"Drift report saved to {report_path.path}")


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name="gpon-predictive-maintenance",
    description="XGBoost-based GPON router failure prediction with MLflow tracking and Evidently drift monitoring.",
)
def gpon_maintenance_pipeline(
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment: str = "gpon-predictive-maintenance",
):
    data_task = generate_data()

    train_task = train_and_log(
        train_dataset=data_task.outputs["train_dataset"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
    )

    drift_task = drift_report(  # noqa: F841
        train_dataset=data_task.outputs["train_dataset"],
        reference_dataset=data_task.outputs["reference_dataset"],
    )
    # Drift report runs in parallel with training
    drift_task.after(data_task)


# ---------------------------------------------------------------------------
# Compile (generates pipeline.yaml for submission to a KFP instance)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "pipeline.yaml")
    kfp.compiler.Compiler().compile(
        pipeline_func=gpon_maintenance_pipeline,
        package_path=output_file,
    )
    print(f"Pipeline compiled → {output_file}")

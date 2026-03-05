"""
MLflow integration: end-to-end training run with experiment tracking,
artifact logging, and model registration for GPON failure prediction.

This module wraps the training and evaluation pipelines inside a single
mlflow.start_run() context.  It uses mlflow.xgboost.autolog() for automatic
parameter/metric capture and additionally logs:
    - Final evaluation metrics (AUC, Precision, Recall, F1)
    - Feature importance plot  (PNG artifact)
    - ROC curve plot            (PNG artifact)
    - Confusion matrix plot     (PNG artifact)
    - The trained XGBoost model to the MLflow Model Registry

Usage (CLI)::::

    python -m src.training.register_model \
        --data-uri      s3://mlflow-artifacts/processed/telemetry.parquet \
        --experiment    gpon-failure-prediction \
        --run-name      xgboost-baseline \
        --model-name    gpon-failure-xgboost \
        --register-model

Environment variables:
    MLFLOW_TRACKING_URI : MLflow server URI  (default: http://localhost:5000)
    MINIO_ENDPOINT      : MinIO endpoint      (default: http://localhost:9000)
    MINIO_ACCESS_KEY    : Access key          (default: minioadmin)
    MINIO_SECRET_KEY    : Secret key          (default: minioadmin)
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv

from src.training.evaluate import (
    EvaluationReport,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from src.training.train_xgboost import (
    TrainingResult,
    chronological_split,
    load_parquet_from_minio,
    train_xgboost,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EXPERIMENT_NAME: str = "gpon-failure-prediction"
DEFAULT_MODEL_NAME: str = "gpon-failure-xgboost"
DEFAULT_ARTIFACT_SUBDIR: str = "plots"


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Set the MLflow tracking URI and ensure the experiment exists.

    Args:
        tracking_uri: URL of the MLflow tracking server.
        experiment_name: Name of the experiment to use (created if missing).
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        logger.info("Created new MLflow experiment: '%s'", experiment_name)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "MLflow configured — tracking URI: %s  experiment: %s",
        tracking_uri,
        experiment_name,
    )


def _log_plots(
    result: TrainingResult,
    report: EvaluationReport,
    artifact_subdir: str = DEFAULT_ARTIFACT_SUBDIR,
) -> None:
    """Generate and log all diagnostic plots as MLflow artifacts.

    Saves each figure to a temporary directory and uploads via
    mlflow.log_artifact().  Temporary files are cleaned up automatically.

    Args:
        result: A completed :class:`TrainingResult`.
        report: Evaluation metrics from :func:`evaluate_model`.
        artifact_subdir: Sub-path within the MLflow artifact store for plots.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Feature importance
        fig_fi = plot_feature_importance(result.model, result.feature_names, top_n=20)
        fi_path = tmp_path / "feature_importance.png"
        fig_fi.savefig(fi_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(fi_path), artifact_path=artifact_subdir)
        fig_fi.clf()
        logger.info("Logged feature importance plot.")

        # 2. ROC curve
        fig_roc = plot_roc_curve(
            result.model,
            result.split.X_test,
            result.split.y_test,
            report,
        )
        roc_path = tmp_path / "roc_curve.png"
        fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(roc_path), artifact_path=artifact_subdir)
        fig_roc.clf()
        logger.info("Logged ROC curve plot.")

        # 3. Confusion matrix
        fig_cm = plot_confusion_matrix(report)
        cm_path = tmp_path / "confusion_matrix.png"
        fig_cm.savefig(cm_path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(str(cm_path), artifact_path=artifact_subdir)
        fig_cm.clf()
        logger.info("Logged confusion matrix plot.")


def _register_model(run_id: str, model_name: str, model_uri: str) -> None:
    """Register the logged model in the MLflow Model Registry.

    If the registered model does not exist it is created automatically.
    The new version is transitioned to the 'Staging' stage immediately after
    registration.

    Args:
        run_id: The MLflow run ID that produced the model.
        model_name: Registered model name in the Model Registry.
        model_uri: URI of the logged model artifact (e.g. ``runs:/<id>/model``).
    """
    client = mlflow.tracking.MlflowClient()

    # Ensure the registered model entry exists
    try:
        client.get_registered_model(model_name)
        logger.info("Registered model '%s' already exists.", model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(
            name=model_name,
            description=(
                "XGBoost binary classifier for GPON router failure prediction. "
                "Predicts Failure_In_7_Days from TR-069 telemetry features."
            ),
            tags={"team": "mlops", "project": "gpon-failure-prediction"},
        )
        logger.info("Created registered model: '%s'", model_name)

    # Register the new version
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
        description=f"Trained in run {run_id}",
    )
    logger.info(
        "Registered model version: %s  (version %s)",
        model_name,
        model_version.version,
    )

    # Transition to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    logger.info(
        "Model '%s' v%s transitioned to Staging.",
        model_name,
        model_version.version,
    )


# ---------------------------------------------------------------------------
# Main MLflow run orchestrator
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    test_size: float = 0.20,
    xgb_params: dict | None = None,
    register: bool = True,
    tracking_uri: str | None = None,
) -> tuple[TrainingResult, EvaluationReport]:
    """Execute the full training pipeline within a single MLflow run.

    Steps performed
    ---------------
    1. Load processed Parquet from MinIO.
    2. Chronological train/test split.
    3. Train XGBoost (with mlflow.xgboost.autolog() capturing params/metrics).
    4. Evaluate on the held-out test set.
    5. Log final evaluation metrics, plots, and the trained model.
    6. Optionally register the model in the MLflow Model Registry.

    Args:
        data_uri: S3 URI of the processed Parquet dataset.
        experiment_name: MLflow experiment name (created if absent).
        run_name: Human-readable name for this run (auto-generated if None).
        model_name: Name under which the model is registered.
        test_size: Fraction of data to use as the test set.
        xgb_params: Override dict for XGBoost hyperparameters.
        register: If True, register the model in the MLflow Model Registry.
        tracking_uri: MLflow tracking server URI (falls back to env var).

    Returns:
        A tuple (TrainingResult, EvaluationReport).

    Raises:
        RuntimeError: If any pipeline step fails.
    """
    effective_tracking_uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    _configure_mlflow(effective_tracking_uri, experiment_name)

    # Enable XGBoost autologging — captures all hyperparameters, training loss
    # curves, and the model artifact automatically.
    mlflow.xgboost.autolog(
        log_input_examples=False,  # disable to avoid large PII artifacts
        log_model_signatures=True,
        log_models=True,
        disable=False,
        exclusive=False,
        registered_model_name=None,  # we handle registration manually below
        silent=False,
    )

    logger.info("=== Starting MLflow Training Run ===")
    logger.info("Experiment  : %s", experiment_name)
    logger.info("Data URI    : %s", data_uri)
    logger.info("Test size   : %.0f%%", test_size * 100)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run started — run_id: %s", run_id)

        try:
            # ----------------------------------------------------------------
            # Step 1: Load data
            # ----------------------------------------------------------------
            df = load_parquet_from_minio(data_uri)
            mlflow.log_param("data_uri", data_uri)
            mlflow.log_param("n_rows_total", len(df))
            mlflow.log_param("n_features_raw", df.shape[1])

            # ----------------------------------------------------------------
            # Step 2: Chronological split
            # ----------------------------------------------------------------
            split = chronological_split(df, test_size=test_size)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("split_timestamp", split.split_timestamp)
            mlflow.log_param("n_train_rows", len(split.X_train))
            mlflow.log_param("n_test_rows", len(split.X_test))
            mlflow.log_param("n_features_engineered", len(split.feature_names))
            mlflow.log_param(
                "positive_rate_train",
                round(float(split.y_train.mean()), 4),
            )

            # ----------------------------------------------------------------
            # Step 3: Train XGBoost
            # ----------------------------------------------------------------
            result = train_xgboost(split, params=xgb_params)
            mlflow.log_param("best_iteration", result.best_iteration)

            # ----------------------------------------------------------------
            # Step 4: Evaluate
            # ----------------------------------------------------------------
            report = evaluate_model(result.model, split.X_test, split.y_test)

            # ----------------------------------------------------------------
            # Step 5: Log evaluation metrics explicitly
            # (autolog captures training-time eval metrics; these are the
            # final held-out test metrics at the optimal threshold)
            # ----------------------------------------------------------------
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in report.to_dict().items()}
            )
            logger.info("Logged final evaluation metrics to MLflow.")

            # ----------------------------------------------------------------
            # Step 6: Log plots as artifacts
            # ----------------------------------------------------------------
            _log_plots(result, report)

            # ----------------------------------------------------------------
            # Step 7: Log feature names as a text artifact for traceability
            # ----------------------------------------------------------------
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as feat_file:
                feat_file.write("\n".join(split.feature_names))
                feat_file_path = feat_file.name
            mlflow.log_artifact(feat_file_path, artifact_path="metadata")
            Path(feat_file_path).unlink(missing_ok=True)

            # ----------------------------------------------------------------
            # Step 8: Register model in the Model Registry
            # ----------------------------------------------------------------
            if register:
                model_uri = f"runs:/{run_id}/model"
                _register_model(run_id=run_id, model_name=model_name, model_uri=model_uri)

            logger.info(
                "=== MLflow Run Complete — run_id: %s  ROC-AUC: %.4f  F1: %.4f ===",
                run_id,
                report.roc_auc,
                report.f1,
            )

        except Exception as exc:
            mlflow.set_tag("run_status", "FAILED")
            mlflow.log_param("failure_reason", str(exc))
            logger.error("Training run FAILED: %s", exc, exc_info=True)
            raise RuntimeError(f"MLflow training pipeline failed: {exc}") from exc

    return result, report


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the registration script.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        Parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and register the GPON failure XGBoost model with MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-uri",
        required=True,
        help="S3 URI of the processed Parquet dataset.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable MLflow run name (auto-generated if omitted).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Registered model name in the MLflow Model Registry.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Fraction of data to use as test set.",
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        default=True,
        help="Register the model in the MLflow Model Registry after training.",
    )
    parser.add_argument(
        "--no-register",
        dest="register_model",
        action="store_false",
        help="Skip model registration.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: parse args and kick off the training pipeline."""
    args = parse_args(argv)

    run_training_pipeline(
        data_uri=args.data_uri,
        experiment_name=args.experiment,
        run_name=args.run_name,
        model_name=args.model_name,
        test_size=args.test_size,
        register=args.register_model,
    )


if __name__ == "__main__":
    main()
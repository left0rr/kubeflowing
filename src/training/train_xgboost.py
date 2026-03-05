"""
XGBoost training pipeline for GPON router failure prediction.

Loads a processed Parquet dataset from MinIO, performs a strict
chronological train/test split (no shuffling — this is time-series data),
trains an XGBoost binary classifier, and returns the fitted model along
with the split DataFrames for downstream evaluation.

Usage (CLI)::

    python -m src.training.train_xgboost \
        --data-uri   s3://mlflow-artifacts/processed/telemetry.parquet \
        --test-size  0.20

Environment variables:
    MINIO_ENDPOINT   : MinIO S3 endpoint URL  (default: http://localhost:9000)
    MINIO_ACCESS_KEY : Access key              (default: minioadmin)
    MINIO_SECRET_KEY : Secret key              (default: minioadmin)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import boto3
import pandas as pd
import xgboost as xgb
from botocore.client import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_COLUMN: str = "Failure_In_7_Days"
TIMESTAMP_COLUMN: str = "Timestamp"
DROP_COLUMNS: list[str] = ["Device_ID", "Timestamp", TARGET_COLUMN]

DEFAULT_XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 10,
    "random_state": 42,
    "n_jobs": -1,
}


dataclass
class TrainTestSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    split_timestamp: str


dataclass
class TrainingResult:
    model: xgb.XGBClassifier
    split: TrainTestSplit
    params: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    best_iteration: int = 0


def _get_s3_client() -> boto3.client:
    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    logger.info("S3/MinIO client created — endpoint: %s", endpoint)
    return client


def load_parquet_from_minio(s3_uri: str) -> pd.DataFrame:
    """Load a Parquet file from MinIO into a pandas DataFrame.

    Args:
        s3_uri: Full S3 URI, e.g. s3://mlflow-artifacts/processed/telemetry.parquet

    Returns:
        Loaded pandas DataFrame.

    Raises:
        ValueError: If the URI is malformed.
        RuntimeError: If the S3 download fails.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with 's3://'): {s3_uri}")
    without_prefix = s3_uri[5:]
    bucket, _, key = without_prefix.partition("/")
    if not bucket or not key:
        raise ValueError(f"Cannot parse bucket/key from URI: {s3_uri}")

    logger.info("Loading Parquet from MinIO: bucket=%s  key=%s", bucket, key)
    client = _get_s3_client()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read()
    except Exception as exc:
        raise RuntimeError(f"Failed to download s3://{bucket}/{key}: {exc}") from exc

    df = pd.read_parquet(io.BytesIO(body), engine="pyarrow")
    logger.info("Loaded dataset: %d rows x %d cols", *df.shape)
    return df


def chronological_split(df: pd.DataFrame, test_size: float = 0.20) -> TrainTestSplit:
    """Split a time-series DataFrame chronologically into train and test sets.

    No shuffling is performed — future data must never appear in the training set.

    Args:
        df: Processed telemetry DataFrame with Timestamp and Failure_In_7_Days.
        test_size: Fraction of rows to reserve for testing (default: 0.20).

    Returns:
        A TrainTestSplit dataclass.

    Raises:
        KeyError: If required columns are missing.
        ValueError: If test_size is not in (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    missing = [c for c in [TIMESTAMP_COLUMN, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for split: {missing}")

    df_sorted = df.sort_values(TIMESTAMP_COLUMN, kind="stable").reset_index(drop=True)
    n_total = len(df_sorted)
    n_test = max(1, int(n_total * test_size))
    n_train = n_total - n_test
    split_timestamp = str(df_sorted.iloc[n_train][TIMESTAMP_COLUMN])

    logger.info(
        "Chronological split — train: %d rows, test: %d rows (boundary: %s)",
        n_train, n_test, split_timestamp,
    )

    feature_cols = [c for c in df_sorted.columns if c not in DROP_COLUMNS]
    train_df = df_sorted.iloc[:n_train]
    test_df = df_sorted.iloc[n_train:]

    logger.info("Train label distribution: %s", train_df[TARGET_COLUMN].value_counts().to_dict())

    return TrainTestSplit(
        X_train=train_df[feature_cols].reset_index(drop=True),
        X_test=test_df[feature_cols].reset_index(drop=True),
        y_train=train_df[TARGET_COLUMN].reset_index(drop=True),
        y_test=test_df[TARGET_COLUMN].reset_index(drop=True),
        feature_names=feature_cols,
        split_timestamp=split_timestamp,
    )


def train_xgboost(split: TrainTestSplit, params: dict | None = None) -> TrainingResult:
    """Train an XGBoost binary classifier on the provided split.

    Args:
        split: A TrainTestSplit from chronological_split().
        params: XGBoost hyperparameters dict. Merged over DEFAULT_XGB_PARAMS.

    Returns:
        A TrainingResult with the fitted model and metadata.

    Raises:
        RuntimeError: If training fails.
    """
    merged_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    early_stopping_rounds = merged_params.pop("early_stopping_rounds", 30)

    logger.info("Training XGBoost with params: %s", merged_params)
    logger.info(
        "Feature matrix shape — train: %s, test: %s",
        split.X_train.shape, split.X_test.shape,
    )

    model = xgb.XGBClassifier(**merged_params)
    try:
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_test, split.y_test)],
            verbose=50,
            early_stopping_rounds=early_stopping_rounds,
        )
    except Exception as exc:
        raise RuntimeError(f"XGBoost training failed: {exc}") from exc

    best_iter = getattr(model, "best_iteration", merged_params.get("n_estimators", 300))
    logger.info("Training complete — best iteration: %d", best_iter)

    return TrainingResult(
        model=model,
        split=split,
        params={**merged_params, "early_stopping_rounds": early_stopping_rounds},
        feature_names=split.feature_names,
        best_iteration=best_iter,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost on processed GPON telemetry from MinIO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-uri", required=True, help="S3 URI of processed Parquet file.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test set fraction.")
    parser.add_argument("--output-dir", default="models/", help="Local directory to save model.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> TrainingResult:
    """Orchestrate the training pipeline from CLI arguments."""
    args = parse_args(argv)
    logger.info("=== GPON XGBoost Training Pipeline ===")
    df = load_parquet_from_minio(args.data_uri)
    split = chronological_split(df, test_size=args.test_size)
    result = train_xgboost(split)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgboost_gpon.json"
    result.model.save_model(str(model_path))
    logger.info("Model saved locally: %s", model_path)
    logger.info("=== Training complete ===")
    return result


if __name__ == "__main__":
    main()
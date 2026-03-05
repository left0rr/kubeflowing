"""
Ingestion script: raw CSV -> validated -> feature-engineered -> MinIO (Parquet).

Usage (CLI):
    python -m src.data.ingest \
        --input-path  data/raw/telemetry_2024_01.csv \
        --output-key  processed/telemetry_2024_01.parquet \
        --bucket      mlflow-artifacts

Environment variables:
    MINIO_ENDPOINT   : MinIO S3 endpoint URL  (default: http://localhost:9000)
    MINIO_ACCESS_KEY : Access key              (default: minioadmin)
    MINIO_SECRET_KEY : Secret key              (default: minioadmin)
    MINIO_BUCKET     : Target bucket name      (default: mlflow-artifacts)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
from botocore.client import Config
from dotenv import load_dotenv

from src.data.feature_engineering import build_feature_matrix
from src.data.validation import validate_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


def _get_minio_client() -> boto3.client:
    """Create and return a boto3 S3 client configured for MinIO."""
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
    logger.info("MinIO client created — endpoint: %s", endpoint)
    return client


def load_raw_csv(input_path: str | Path) -> pd.DataFrame:
    """Load a raw telemetry CSV from the local filesystem.

    Args:
        input_path: Path to the CSV file.

    Returns:
        Raw pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV is empty.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path.resolve()}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {path.resolve()}")
    logger.info("Loaded raw CSV: %s — %d rows x %d cols", path.name, *df.shape)
    return df


def run_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Validate every row via the Pydantic schema; drop invalid rows.

    Args:
        df: Raw telemetry DataFrame.

    Returns:
        Cleaned DataFrame with only valid rows.

    Raises:
        SystemExit: If zero rows pass validation.
    """
    records = df.to_dict(orient="records")
    valid_models, rejected = validate_records(records)
    rejection_rate = len(rejected) / max(len(records), 1) * 100
    logger.info("Validation: %d/%d rows accepted (%.1f%% rejection rate).", len(valid_models), len(records), rejection_rate)
    if not valid_models:
        logger.error("All rows failed validation — aborting ingestion.")
        sys.exit(1)
    if rejection_rate > 20.0:
        logger.warning("High rejection rate (%.1f%%) — check upstream data quality.", rejection_rate)
    clean_df = pd.DataFrame([m.model_dump() for m in valid_models])
    return clean_df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline.

    Args:
        df: Validated telemetry DataFrame.

    Returns:
        Fully-engineered DataFrame.
    """
    engineered = build_feature_matrix(df)
    logger.info("Feature engineering complete: %d rows x %d cols.", *engineered.shape)
    return engineered


def upload_to_minio(df: pd.DataFrame, bucket: str, object_key: str, client: boto3.client) -> str:
    """Serialise df to Parquet and upload to MinIO.

    Args:
        df: Processed feature DataFrame.
        bucket: Target S3/MinIO bucket name.
        object_key: Object key within the bucket.
        client: Configured boto3 S3 client.

    Returns:
        Full S3 URI of the uploaded object.
    """
    existing_buckets = [b["Name"] for b in client.list_buckets().get("Buckets", [])]
    if bucket not in existing_buckets:
        client.create_bucket(Bucket=bucket)
        logger.info("Created bucket: %s", bucket)

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow", compression="snappy")
    buffer.seek(0)

    client.upload_fileobj(
        Fileobj=buffer,
        Bucket=bucket,
        Key=object_key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    s3_uri = f"s3://{bucket}/{object_key}"
    logger.info("Uploaded processed dataset -> %s", s3_uri)
    return s3_uri


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest raw GPON telemetry CSV, validate, engineer features, and upload to MinIO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-path", required=True, help="Path to the raw telemetry CSV file.")
    parser.add_argument("--output-key", required=True, help="S3 object key for the output Parquet file.")
    parser.add_argument("--bucket", default=os.getenv("MINIO_BUCKET", "mlflow-artifacts"), help="MinIO bucket name.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Orchestrate the full ingestion pipeline."""
    args = parse_args(argv)
    logger.info("=== GPON Telemetry Ingestion Pipeline ===")
    logger.info("Input  : %s", args.input_path)
    logger.info("Output : s3://%s/%s", args.bucket, args.output_key)

    raw_df = load_raw_csv(args.input_path)
    clean_df = run_validation(raw_df)
    engineered_df = run_feature_engineering(clean_df)

    s3_client = _get_minio_client()
    s3_uri = upload_to_minio(df=engineered_df, bucket=args.bucket, object_key=args.output_key, client=s3_client)
    logger.info("=== Ingestion complete: %s ===", s3_uri)


if __name__ == "__main__":
    main()

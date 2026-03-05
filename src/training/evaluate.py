"""
Model evaluation functions for GPON router failure prediction.

Provides utilities to compute classification metrics (AUC, Precision, Recall,
F1-score), generate a confusion matrix, and log all results as a structured
EvaluationReport dataclass that is ready to be passed to the MLflow logging layer.

All functions are pure — they take arrays/DataFrames in and return plain Python
objects out, with no side effects (no MLflow calls here).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # non-interactive backend — safe in containers and CI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """Structured container for all classification evaluation metrics.

    Attributes:
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the Precision-Recall curve (better for imbalanced data).
        precision: Precision at the chosen decision threshold.
        recall: Recall (sensitivity) at the chosen decision threshold.
        f1: F1-score at the chosen decision threshold.
        threshold: Decision threshold used to convert probabilities to labels.
        support_positive: Number of positive samples in the evaluation set.
        support_total: Total number of samples in the evaluation set.
        confusion: Confusion matrix as a nested list [[TN, FP], [FN, TP]].
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    support_positive: int
    support_total: int
    confusion: list[list[int]]

    def to_dict(self) -> dict:
        """Return metrics as a flat dict suitable for mlflow.log_metrics().

        Returns:
            Dict of metric names to float/int values (confusion matrix excluded).
        """
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "support_positive": float(self.support_positive),
            "support_total": float(self.support_total),
        }


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------


def compute_optimal_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Find the decision threshold that maximises the F1-score.

    For imbalanced telecom failure datasets the default threshold of 0.5 is
    often sub-optimal.  This function scans the PR curve to find the threshold
    that yields the best F1 at inference time.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Optimal threshold in [0, 1].
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid division by zero
    f1_scores = np.where(
        (precisions + recalls) == 0,
        0.0,
        2 * (precisions * recalls) / (precisions + recalls),
    )
    best_idx = int(np.argmax(f1_scores[:-1]))  # thresholds has one fewer element
    optimal = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold: %.4f  (F1=%.4f at that threshold)",
        optimal,
        f1_scores[best_idx],
    )
    return optimal


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    threshold: float | None = None,
) -> EvaluationReport:
    """Evaluate a trained XGBoost model on the test set.

    Computes ROC-AUC, PR-AUC, Precision, Recall, and F1 at an optionally
    provided (or auto-derived) decision threshold.

    Args:
        model: A fitted :class:`xgboost.XGBClassifier`.
        X_test: Test feature matrix.
        y_test: Ground-truth binary labels for the test set.
        threshold: Decision threshold.  If ``None``, the threshold that
            maximises F1 on the test set is used automatically.

    Returns:
        An :class:`EvaluationReport` with all computed metrics.

    Raises:
        ValueError: If *X_test* and *y_test* have incompatible lengths.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"X_test ({len(X_test)} rows) and y_test ({len(y_test)} rows) must have the same length."
        )

    y_true = np.asarray(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    if threshold is None:
        threshold = compute_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    report = EvaluationReport(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        support_positive=int(y_true.sum()),
        support_total=len(y_true),
        confusion=cm,
    )

    logger.info(
        "Evaluation results — ROC-AUC: %.4f  PR-AUC: %.4f  "
        "Precision: %.4f  Recall: %.4f  F1: %.4f  (threshold=%.4f)",
        roc_auc, pr_auc, precision, recall, f1, threshold,
    )
    logger.info("Confusion matrix:\n%s", np.array(cm))
    logger.info(
        "\nClassification report:\n%s",
        classification_report(y_true, y_pred, target_names=["No Failure", "Failure"], zero_division=0),
    )

    return report


# ---------------------------------------------------------------------------
# Plot generation helpers
# ---------------------------------------------------------------------------


def plot_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> plt.Figure:
    """Generate a horizontal bar chart of the top-N XGBoost feature importances.

    Uses the ``weight`` (split count) importance type by default, which is the
    most interpretable for business stakeholders.

    Args:
        model: A fitted :class:`xgboost.XGBClassifier`.
        feature_names: Ordered list of feature names matching the training columns.
        top_n: Number of top features to display (default: 20).

    Returns:
        A :class:`matplotlib.figure.Figure` ready for saving or MLflow logging.
    """
    booster = model.get_booster()
    importance_raw = booster.get_score(importance_type="weight")

    # Map internal f0, f1, ... names to human-readable feature names
    importance_named = {
        feature_names[int(k[1:])]: v
        for k, v in importance_raw.items()
        if k[1:].isdigit() and int(k[1:]) < len(feature_names)
    }
    # Fall back to get_feature_importances_ if booster score is empty
    if not importance_named:
        importances = model.feature_importances_
        importance_named = dict(zip(feature_names, importances.tolist()))

    series = (
        pd.Series(importance_named, name="Importance")
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    series.plot(kind="barh", ax=ax, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Feature Importance (weight)", fontsize=12)
    ax.set_title(f"XGBoost Top-{top_n} Feature Importances — GPON Failure Prediction", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    logger.debug("Feature importance plot generated (%d features shown).", len(series))
    return fig


def plot_roc_curve(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    report: EvaluationReport,
) -> plt.Figure:
    """Generate an ROC curve plot with the operating point marked.

    Args:
        model: Fitted XGBoost classifier.
        X_test: Test feature matrix.
        y_test: Ground-truth labels.
        report: An EvaluationReport (used to mark the operating threshold point).

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    y_true = np.asarray(y_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"ROC (AUC = {report.roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1)
    ax.scatter(
        [1 - report.precision], [report.recall],
        marker="o", color="#FF9800", zorder=5,
        label=f"Operating point (thr={report.threshold:.3f})",
    )
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — GPON Router Failure Prediction", fontsize=13)
    ax.legend(loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    report: EvaluationReport,
) -> plt.Figure:
    """Generate a confusion matrix heatmap from an EvaluationReport.

    Args:
        report: An :class:`EvaluationReport` containing the confusion matrix.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    cm = np.array(report.confusion)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — GPON Failure Prediction", fontsize=12)
    plt.tight_layout()
    return fig
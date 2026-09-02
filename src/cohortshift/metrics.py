"""Classification metrics with explicit probability calibration."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class FoldMetrics:
    fold: int
    train_rows: int
    test_rows: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    matthews_correlation: float
    log_loss: float
    expected_calibration_error: float
    roc_auc: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_calibration_error(
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    confidence = probabilities.max(axis=1)
    correct = (predictions == truth).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    if total == 0:
        raise ValueError("cannot calculate calibration on an empty array")
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence > lower) & (confidence <= upper)
        count = int(selected.sum())
        if count:
            calibration_gap = float(correct[selected].mean() - confidence[selected].mean())
            error += count / total * abs(calibration_gap)
    return error


def classification_metrics(
    *,
    fold: int,
    train_rows: int,
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> FoldMetrics:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        balanced = float(balanced_accuracy_score(truth, predictions))
        macro_f1 = float(f1_score(truth, predictions, average="macro", zero_division=0))
        matthews = float(matthews_corrcoef(truth, predictions))
    auc: float | None = None
    if len(np.unique(truth)) == len(classes):
        try:
            if len(classes) == 2:
                positive_truth = (truth == classes[1]).astype(int)
                auc = float(roc_auc_score(positive_truth, probabilities[:, 1]))
            else:
                auc = float(
                    roc_auc_score(
                        truth,
                        probabilities,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except ValueError:
            auc = None
    return FoldMetrics(
        fold=fold,
        train_rows=train_rows,
        test_rows=len(truth),
        accuracy=float(accuracy_score(truth, predictions)),
        balanced_accuracy=balanced,
        macro_f1=macro_f1,
        matthews_correlation=matthews,
        log_loss=float(log_loss(truth, probabilities, labels=classes)),
        expected_calibration_error=expected_calibration_error(truth, predictions, probabilities),
        roc_auc=auc,
    )


def aggregate_metrics(folds: list[FoldMetrics]) -> dict[str, dict[str, float | int | None]]:
    if not folds:
        raise ValueError("cannot aggregate an empty fold list")
    output: dict[str, dict[str, float | int | None]] = {}
    names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "matthews_correlation",
        "log_loss",
        "expected_calibration_error",
        "roc_auc",
    )
    for name in names:
        values = [getattr(fold, name) for fold in folds]
        present = np.array([value for value in values if value is not None], dtype=float)
        output[name] = {
            "mean": float(present.mean()) if len(present) else None,
            "standard_deviation": float(present.std(ddof=0)) if len(present) else None,
            "observed_folds": len(present),
        }
    return output

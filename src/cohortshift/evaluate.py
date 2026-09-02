"""Leakage-resistant model evaluation and artifact generation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cohortshift.audit import audit_frame
from cohortshift.config import EvaluationConfig
from cohortshift.drift import DriftRecord, measure_drift
from cohortshift.metrics import FoldMetrics, aggregate_metrics, classification_metrics
from cohortshift.split import TemporalFold, temporal_folds


class EvaluationError(RuntimeError):
    """Raised when an evaluation cannot satisfy its declared contract."""


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    output_directory: str
    report: str
    metrics: str
    drift: str
    predictions: str
    folds: int
    rows: int


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_digest(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(columns).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(frame.loc[:, list(columns)], index=False)
    digest.update(hashes.to_numpy(dtype=np.uint64).tobytes())
    return digest.hexdigest()


def _pipeline(config: EvaluationConfig) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    numeric = list(config.numeric_columns())
    categorical = list(config.categorical_columns)
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    (
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    )
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    (
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    )
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop")
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=config.random_seed,
    )
    return Pipeline((("preprocess", preprocess), ("classifier", classifier)))


def _feature_importance(model: Pipeline) -> dict[str, float]:
    preprocess = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    names = preprocess.get_feature_names_out()
    coefficients = np.asarray(classifier.coef_, dtype=float)
    importance = np.abs(coefficients).mean(axis=0)
    return {str(name): float(value) for name, value in zip(names, importance, strict=True)}


def _prediction_rows(
    frame: pd.DataFrame,
    fold: TemporalFold,
    config: EvaluationConfig,
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test = frame.iloc[fold.test_positions]
    for position, (_, source_row), actual, predicted, probability in zip(
        fold.test_positions,
        test.iterrows(),
        truth,
        predictions,
        probabilities,
        strict=True,
    ):
        rows.append(
            {
                "fold": fold.number,
                "row_position": int(position),
                "time": str(source_row[config.time_column]),
                "actual": str(actual),
                "predicted": str(predicted),
                "probabilities": json.dumps(
                    {
                        str(label): float(value)
                        for label, value in zip(classes, probability, strict=True)
                    },
                    sort_keys=True,
                ),
            }
        )
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def evaluate_frame(
    frame: pd.DataFrame,
    config: EvaluationConfig,
    output_directory: Path,
    *,
    source_name: str = "in-memory",
    source_sha256: str | None = None,
) -> EvaluationResult:
    if frame.empty:
        raise EvaluationError("dataset is empty")
    findings = audit_frame(frame, config)
    audit_errors = [finding for finding in findings if finding.severity == "error"]
    structural_codes = {
        "invalid-time",
        "missing-column",
        "missing-target",
        "missing-time",
        "non-numeric-feature",
    }
    has_structural_error = any(finding.code in structural_codes for finding in audit_errors)
    if audit_errors and (config.fail_on_audit_errors or has_structural_error):
        codes = ", ".join(sorted({finding.code for finding in audit_errors}))
        raise EvaluationError(f"dataset audit failed: {codes}")
    try:
        folds = temporal_folds(frame, config)
    except ValueError as error:
        raise EvaluationError(f"cannot construct temporal folds: {error}") from error

    fold_metrics: list[FoldMetrics] = []
    drift_records: list[DriftRecord] = []
    prediction_rows: list[dict[str, Any]] = []
    feature_importances: defaultdict[str, list[float]] = defaultdict(list)

    for fold in folds:
        train = frame.iloc[fold.train_positions]
        test = frame.iloc[fold.test_positions]
        train_target = train[config.target_column].astype(str).to_numpy()
        test_target = test[config.target_column].astype(str).to_numpy()
        if len(np.unique(train_target)) < 2:
            raise EvaluationError(f"fold {fold.number} training target has fewer than two classes")
        unseen_classes = set(test_target) - set(train_target)
        if unseen_classes:
            labels = ", ".join(sorted(unseen_classes))
            raise EvaluationError(f"fold {fold.number} test target has unseen classes: {labels}")
        model = _pipeline(config)
        try:
            model.fit(train.loc[:, list(config.feature_columns)], train_target)
            predictions = np.asarray(
                model.predict(test.loc[:, list(config.feature_columns)]), dtype=str
            )
            probabilities = np.asarray(
                model.predict_proba(test.loc[:, list(config.feature_columns)]), dtype=float
            )
        except (TypeError, ValueError) as error:
            raise EvaluationError(f"fold {fold.number} model fitting failed: {error}") from error
        classes = np.asarray(model.named_steps["classifier"].classes_, dtype=str)
        fold_metrics.append(
            classification_metrics(
                fold=fold.number,
                train_rows=fold.train_rows,
                truth=test_target,
                predictions=predictions,
                probabilities=probabilities,
                classes=classes,
            )
        )
        drift_records.extend(
            measure_drift(
                train,
                test,
                features=config.feature_columns,
                categorical=config.categorical_columns,
                fold=fold.number,
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                frame,
                fold,
                config,
                test_target,
                predictions,
                probabilities,
                classes,
            )
        )
        for feature, importance in _feature_importance(model).items():
            feature_importances[feature].append(importance)

    output = output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "fold_metrics.csv"
    drift_path = output / "drift.csv"
    predictions_path = output / "predictions.csv"
    audit_path = output / "audit.csv"
    report_path = output / "report.json"
    _atomic_frame(metrics_path, pd.DataFrame([metric.as_dict() for metric in fold_metrics]))
    _atomic_frame(drift_path, pd.DataFrame([record.as_dict() for record in drift_records]))
    _atomic_frame(
        audit_path,
        pd.DataFrame(
            [finding.as_dict() for finding in findings],
            columns=["code", "severity", "column", "detail"],
        ),
    )
    _atomic_frame(predictions_path, pd.DataFrame(prediction_rows))

    average_importance: list[dict[str, Any]] = [
        {"feature": feature, "mean_absolute_coefficient": float(np.mean(values))}
        for feature, values in feature_importances.items()
    ]
    average_importance.sort(
        key=lambda item: (
            -float(item["mean_absolute_coefficient"]),
            str(item["feature"]),
        )
    )
    report = {
        "schema_version": 1,
        "source": {
            "name": source_name,
            "sha256": source_sha256,
            "rows": len(frame),
            "frame_sha256": _frame_digest(frame, config.required_columns()),
        },
        "configuration": config.as_dict(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "audit": [finding.as_dict() for finding in findings],
        "folds": [fold.as_dict() for fold in folds],
        "fold_metrics": [metric.as_dict() for metric in fold_metrics],
        "aggregate_metrics": aggregate_metrics(fold_metrics),
        "feature_importance": average_importance,
        "drift_summary": {
            "stable": sum(record.level == "stable" for record in drift_records),
            "moderate": sum(record.level == "moderate" for record in drift_records),
            "high": sum(record.level == "high" for record in drift_records),
        },
    }
    _atomic_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return EvaluationResult(
        output_directory=str(output),
        report=str(report_path),
        metrics=str(metrics_path),
        drift=str(drift_path),
        predictions=str(predictions_path),
        folds=len(folds),
        rows=len(frame),
    )


def evaluate_csv(
    csv_path: Path,
    config: EvaluationConfig,
    output_directory: Path,
) -> EvaluationResult:
    source = csv_path.expanduser().resolve()
    if not source.is_file():
        raise EvaluationError(f"CSV does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise EvaluationError(f"cannot read CSV: {error}") from error
    return evaluate_frame(
        frame,
        config,
        output_directory,
        source_name=source.name,
        source_sha256=_file_digest(source),
    )

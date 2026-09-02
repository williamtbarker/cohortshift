"""Pre-evaluation schema and leakage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from cohortshift.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: str
    column: str | None
    detail: str

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _exact_target_copy(feature: pd.Series, target: pd.Series) -> bool:
    present = feature.notna() & target.notna()
    if not bool(present.any()):
        return False
    return bool(feature[present].astype(str).equals(target[present].astype(str)))


def audit_frame(frame: pd.DataFrame, config: EvaluationConfig) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    missing_columns = [column for column in config.required_columns() if column not in frame]
    for column in missing_columns:
        findings.append(
            AuditFinding("missing-column", "error", column, "required column is absent")
        )
    if missing_columns:
        return findings

    missing_time = int(frame[config.time_column].isna().sum())
    if missing_time:
        findings.append(
            AuditFinding(
                "missing-time",
                "error",
                config.time_column,
                f"{missing_time} rows have no time value",
            )
        )
    parsed_time = pd.to_datetime(
        frame[config.time_column], errors="coerce", utc=True, format="mixed"
    )
    invalid_time = int((parsed_time.isna() & frame[config.time_column].notna()).sum())
    if invalid_time:
        findings.append(
            AuditFinding(
                "invalid-time",
                "error",
                config.time_column,
                f"{invalid_time} rows have invalid time values",
            )
        )
    missing_target = int(frame[config.target_column].isna().sum())
    if missing_target:
        findings.append(
            AuditFinding(
                "missing-target",
                "error",
                config.target_column,
                f"{missing_target} rows have no target value",
            )
        )
    duplicate_rows = int(frame.duplicated(subset=list(config.required_columns())).sum())
    if duplicate_rows:
        findings.append(
            AuditFinding(
                "duplicate-rows",
                "warning",
                None,
                f"{duplicate_rows} rows duplicate all required fields",
            )
        )

    target = frame[config.target_column]
    for column in config.feature_columns:
        series = frame[column]
        if _exact_target_copy(series, target):
            findings.append(
                AuditFinding(
                    "target-copy",
                    "error",
                    column,
                    "feature exactly reproduces the target where both are present",
                )
            )
        missing_fraction = float(series.isna().mean())
        if missing_fraction >= 0.4:
            findings.append(
                AuditFinding(
                    "high-missingness",
                    "warning",
                    column,
                    f"{missing_fraction:.1%} of values are missing",
                )
            )
        is_constant = series.nunique(dropna=True) <= 1
        if is_constant:
            findings.append(
                AuditFinding("constant-feature", "warning", column, "feature is constant")
            )
        if column in config.categorical_columns:
            nonmissing = int(series.notna().sum())
            cardinality = int(series.nunique(dropna=True))
            if nonmissing >= 20 and cardinality / nonmissing > 0.8:
                findings.append(
                    AuditFinding(
                        "high-cardinality-category",
                        "warning",
                        column,
                        f"{cardinality} categories across {nonmissing} populated rows",
                    )
                )
        elif not is_numeric_dtype(series):
            findings.append(
                AuditFinding(
                    "non-numeric-feature",
                    "error",
                    column,
                    "feature must be numeric or declared categorical",
                )
            )
        elif is_numeric_dtype(target) and not is_constant:
            paired = pd.DataFrame({"feature": series, "target": target}).dropna()
            if len(paired) >= 3:
                correlation = float(np.corrcoef(paired["feature"], paired["target"])[0, 1])
                if np.isfinite(correlation) and abs(correlation) >= 0.995:
                    findings.append(
                        AuditFinding(
                            "near-target-correlation",
                            "warning",
                            column,
                            f"absolute target correlation is {abs(correlation):.4f}",
                        )
                    )
    return findings

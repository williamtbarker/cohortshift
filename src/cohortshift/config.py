"""Validated evaluation configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SUPPORTED_FREQUENCIES = frozenset({"Y", "Q", "M", "W", "D"})


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    time_column: str
    target_column: str
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ()
    group_column: str | None = None
    frequency: str = "Y"
    folds: int = 3
    test_periods: int = 1
    gap_periods: int = 0
    minimum_train_periods: int = 2
    random_seed: int = 42
    fail_on_audit_errors: bool = True

    def __post_init__(self) -> None:
        if not self.time_column.strip() or not self.target_column.strip():
            raise ValueError("time_column and target_column are required")
        if not self.feature_columns:
            raise ValueError("at least one feature column is required")
        if any(not column.strip() for column in self.feature_columns):
            raise ValueError("feature column names cannot be blank")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature columns must be unique")
        forbidden = {self.time_column, self.target_column}
        if self.group_column is not None:
            forbidden.add(self.group_column)
        overlap = forbidden.intersection(self.feature_columns)
        if overlap:
            raise ValueError(f"metadata columns cannot be features: {', '.join(sorted(overlap))}")
        unknown_categorical = set(self.categorical_columns) - set(self.feature_columns)
        if unknown_categorical:
            raise ValueError(
                "categorical columns must also be features: "
                + ", ".join(sorted(unknown_categorical))
            )
        normalized_frequency = self.frequency.upper()
        if normalized_frequency not in SUPPORTED_FREQUENCIES:
            raise ValueError(
                f"frequency must be one of: {', '.join(sorted(SUPPORTED_FREQUENCIES))}"
            )
        object.__setattr__(self, "frequency", normalized_frequency)
        if self.folds < 1:
            raise ValueError("folds must be positive")
        if self.test_periods < 1:
            raise ValueError("test_periods must be positive")
        if self.gap_periods < 0:
            raise ValueError("gap_periods cannot be negative")
        if self.minimum_train_periods < 1:
            raise ValueError("minimum_train_periods must be positive")

    def required_columns(self) -> tuple[str, ...]:
        columns = [self.time_column, self.target_column, *self.feature_columns]
        if self.group_column is not None:
            columns.append(self.group_column)
        return tuple(dict.fromkeys(columns))

    def numeric_columns(self) -> tuple[str, ...]:
        categorical = set(self.categorical_columns)
        return tuple(column for column in self.feature_columns if column not in categorical)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

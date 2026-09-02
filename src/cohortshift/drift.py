"""Portable numeric and categorical drift measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class DriftRecord:
    fold: int
    feature: str
    kind: str
    value: float
    level: str

    def as_dict(self) -> dict[str, int | str | float]:
        return asdict(self)


def _normalized(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    adjusted = values.astype(float) + epsilon
    return adjusted / adjusted.sum()


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    reference_numeric = pd.to_numeric(reference, errors="coerce")
    current_numeric = pd.to_numeric(current, errors="coerce")
    reference_present = reference_numeric.dropna().to_numpy(dtype=float)
    current_present = current_numeric.dropna().to_numpy(dtype=float)
    if not len(reference_present) or not len(current_present):
        return 0.0 if not len(reference_present) and not len(current_present) else float("inf")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference_present, quantiles))
    if len(edges) == 1:
        center = edges[0]
        edges = np.array(
            [
                -np.inf,
                np.nextafter(center, -np.inf),
                np.nextafter(center, np.inf),
                np.inf,
            ]
        )
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf
    reference_counts = np.histogram(reference_present, bins=edges)[0].astype(float)
    current_counts = np.histogram(current_present, bins=edges)[0].astype(float)
    reference_counts = np.append(reference_counts, reference_numeric.isna().sum())
    current_counts = np.append(current_counts, current_numeric.isna().sum())
    reference_distribution = _normalized(reference_counts)
    current_distribution = _normalized(current_counts)
    return float(
        np.sum(
            (current_distribution - reference_distribution)
            * np.log(current_distribution / reference_distribution)
        )
    )


def categorical_js_divergence(reference: pd.Series, current: pd.Series) -> float:
    reference_text = reference.astype("string").fillna("[missing]")
    current_text = current.astype("string").fillna("[missing]")
    categories = sorted(set(reference_text).union(current_text))
    if not categories:
        return 0.0
    reference_counts = reference_text.value_counts().reindex(categories, fill_value=0).to_numpy()
    current_counts = current_text.value_counts().reindex(categories, fill_value=0).to_numpy()
    reference_distribution = _normalized(reference_counts)
    current_distribution = _normalized(current_counts)
    midpoint = 0.5 * (reference_distribution + current_distribution)
    divergence = 0.5 * np.sum(reference_distribution * np.log(reference_distribution / midpoint))
    divergence += 0.5 * np.sum(current_distribution * np.log(current_distribution / midpoint))
    return float(divergence)


def _level(kind: str, value: float) -> str:
    if not np.isfinite(value):
        return "high"
    if kind == "numeric-psi":
        return "stable" if value < 0.1 else "moderate" if value < 0.25 else "high"
    return "stable" if value < 0.05 else "moderate" if value < 0.15 else "high"


def measure_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: tuple[str, ...],
    categorical: tuple[str, ...],
    fold: int,
) -> list[DriftRecord]:
    categorical_set = set(categorical)
    records: list[DriftRecord] = []
    for feature in features:
        if feature in categorical_set:
            kind = "categorical-js"
            value = categorical_js_divergence(train[feature], test[feature])
        else:
            kind = "numeric-psi"
            value = population_stability_index(train[feature], test[feature])
        records.append(DriftRecord(fold, feature, kind, value, _level(kind, value)))
    return records

"""Expanding-window temporal splits with optional repeated-entity purging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cohortshift.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class TemporalFold:
    number: int
    train_positions: np.ndarray
    test_positions: np.ndarray
    train_period_start: str
    train_period_end: str
    test_period_start: str
    test_period_end: str
    train_rows: int
    test_rows: int
    purged_train_rows: int
    overlapping_groups: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "number": self.number,
            "train_period_start": self.train_period_start,
            "train_period_end": self.train_period_end,
            "test_period_start": self.test_period_start,
            "test_period_end": self.test_period_end,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "purged_train_rows": self.purged_train_rows,
            "overlapping_groups": self.overlapping_groups,
        }


def _periods(values: pd.Series, frequency: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
    if timestamps.isna().any():
        invalid = int(timestamps.isna().sum())
        raise ValueError(f"time column contains {invalid} missing or invalid values")
    return timestamps.dt.tz_localize(None).dt.to_period(frequency)


def temporal_folds(frame: pd.DataFrame, config: EvaluationConfig) -> list[TemporalFold]:
    if config.time_column not in frame:
        raise ValueError(f"missing time column: {config.time_column}")
    periods = _periods(frame[config.time_column], config.frequency)
    unique_periods = sorted(periods.unique())
    required = (
        config.minimum_train_periods + config.gap_periods + config.folds * config.test_periods
    )
    if len(unique_periods) < required:
        raise ValueError(
            f"need at least {required} periods for this split configuration; "
            f"found {len(unique_periods)}"
        )

    first_test = len(unique_periods) - config.folds * config.test_periods
    folds: list[TemporalFold] = []
    for offset in range(config.folds):
        test_start = first_test + offset * config.test_periods
        test_end = test_start + config.test_periods
        train_end = test_start - config.gap_periods
        train_periods = unique_periods[:train_end]
        test_periods = unique_periods[test_start:test_end]
        # pandas 3 may expose read-only arrays under copy-on-write.
        train_mask = periods.isin(train_periods).to_numpy(dtype=bool, copy=True)
        test_mask = periods.isin(test_periods).to_numpy(dtype=bool, copy=True)
        overlapping_groups = 0
        purged_rows = 0

        if config.group_column is not None:
            if config.group_column not in frame:
                raise ValueError(f"missing group column: {config.group_column}")
            train_groups = set(frame.loc[train_mask, config.group_column].dropna())
            test_groups = set(frame.loc[test_mask, config.group_column].dropna())
            overlap = train_groups.intersection(test_groups)
            overlapping_groups = len(overlap)
            if overlap:
                purge_mask = train_mask & frame[config.group_column].isin(overlap).to_numpy()
                purged_rows = int(purge_mask.sum())
                train_mask &= ~purge_mask

        train_positions = np.flatnonzero(train_mask)
        test_positions = np.flatnonzero(test_mask)
        if not len(train_positions) or not len(test_positions):
            raise ValueError(f"fold {offset + 1} has an empty train or test partition")
        folds.append(
            TemporalFold(
                number=offset + 1,
                train_positions=train_positions,
                test_positions=test_positions,
                train_period_start=str(train_periods[0]),
                train_period_end=str(train_periods[-1]),
                test_period_start=str(test_periods[0]),
                test_period_end=str(test_periods[-1]),
                train_rows=len(train_positions),
                test_rows=len(test_positions),
                purged_train_rows=purged_rows,
                overlapping_groups=overlapping_groups,
            )
        )
    return folds

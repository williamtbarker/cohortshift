from __future__ import annotations

import numpy as np
import pandas as pd

from cohortshift.config import EvaluationConfig


def make_frame(
    *,
    first_year: int = 2015,
    years: int = 10,
    rows_per_year: int = 24,
    repeated_groups: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for year_offset in range(years):
        year = first_year + year_offset
        for row_number in range(rows_per_year):
            target = int((row_number + year_offset) % 3 != 0)
            rows.append(
                {
                    "sample_date": f"{year}-{row_number % 12 + 1:02d}-15",
                    "outcome": target,
                    "signal": target * 1.4 + year_offset * 0.08 + rng.normal(0, 0.45),
                    "noise": rng.normal(0, 1),
                    "site": "north" if row_number % 2 else "south",
                    "entity_id": (
                        f"entity-repeated-{row_number}"
                        if repeated_groups and row_number < 6
                        else f"entity-{year}-{row_number}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_config(**overrides: object) -> EvaluationConfig:
    values: dict[str, object] = {
        "time_column": "sample_date",
        "target_column": "outcome",
        "feature_columns": ("signal", "noise", "site"),
        "categorical_columns": ("site",),
        "group_column": "entity_id",
        "frequency": "Y",
        "folds": 3,
        "test_periods": 1,
        "minimum_train_periods": 3,
        "random_seed": 42,
    }
    values.update(overrides)
    return EvaluationConfig(**values)  # type: ignore[arg-type]

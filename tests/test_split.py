from __future__ import annotations

import unittest

import pandas as pd

from cohortshift.split import temporal_folds
from tests.helpers import make_config, make_frame


class SplitTests(unittest.TestCase):
    def test_expanding_folds_never_train_on_future_rows(self) -> None:
        frame = make_frame()
        folds = temporal_folds(frame, make_config())
        self.assertEqual(len(folds), 3)
        for fold in folds:
            train_dates = pd.to_datetime(frame.iloc[fold.train_positions]["sample_date"])
            test_dates = pd.to_datetime(frame.iloc[fold.test_positions]["sample_date"])
            self.assertLess(train_dates.max(), test_dates.min())
        self.assertLess(folds[0].train_rows, folds[-1].train_rows)

    def test_gap_excludes_period_before_test(self) -> None:
        frame = make_frame()
        folds = temporal_folds(frame, make_config(gap_periods=1, folds=2))
        first = folds[0]
        train_years = set(pd.to_datetime(frame.iloc[first.train_positions]["sample_date"]).dt.year)
        test_year = pd.to_datetime(frame.iloc[first.test_positions]["sample_date"]).dt.year.min()
        self.assertNotIn(test_year - 1, train_years)

    def test_repeated_entities_are_purged_from_training(self) -> None:
        frame = make_frame(repeated_groups=True)
        folds = temporal_folds(frame, make_config())
        for fold in folds:
            train_groups = set(frame.iloc[fold.train_positions]["entity_id"])
            test_groups = set(frame.iloc[fold.test_positions]["entity_id"])
            self.assertFalse(train_groups.intersection(test_groups))
            self.assertGreater(fold.purged_train_rows, 0)

    def test_requires_enough_periods(self) -> None:
        with self.assertRaisesRegex(ValueError, "need at least"):
            temporal_folds(make_frame(years=4), make_config(folds=3))

    def test_rejects_invalid_time_values(self) -> None:
        frame = make_frame()
        frame.loc[0, "sample_date"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "invalid"):
            temporal_folds(frame, make_config())


if __name__ == "__main__":
    unittest.main()

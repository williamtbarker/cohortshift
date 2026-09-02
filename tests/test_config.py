from __future__ import annotations

import unittest

from cohortshift.config import EvaluationConfig


class ConfigTests(unittest.TestCase):
    def test_normalizes_frequency_and_derives_numeric_columns(self) -> None:
        config = EvaluationConfig(
            "date",
            "target",
            ("age", "site"),
            ("site",),
            frequency="m",
        )
        self.assertEqual(config.frequency, "M")
        self.assertEqual(config.numeric_columns(), ("age",))

    def test_rejects_target_as_feature(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata columns"):
            EvaluationConfig("date", "target", ("target",))

    def test_rejects_unknown_categorical_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "categorical columns"):
            EvaluationConfig("date", "target", ("age",), ("site",))

    def test_rejects_invalid_split_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "folds"):
            EvaluationConfig("date", "target", ("age",), folds=0)
        with self.assertRaisesRegex(ValueError, "gap"):
            EvaluationConfig("date", "target", ("age",), gap_periods=-1)


if __name__ == "__main__":
    unittest.main()

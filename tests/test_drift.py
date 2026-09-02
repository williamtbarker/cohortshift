from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cohortshift.drift import (
    categorical_js_divergence,
    measure_drift,
    population_stability_index,
)


class DriftTests(unittest.TestCase):
    def test_identical_numeric_distributions_have_zero_psi(self) -> None:
        values = pd.Series(np.arange(100, dtype=float))
        self.assertAlmostEqual(population_stability_index(values, values), 0.0)

    def test_shifted_numeric_distribution_has_positive_psi(self) -> None:
        reference = pd.Series(np.linspace(0, 1, 200))
        current = pd.Series(np.linspace(2, 3, 200))
        self.assertGreater(population_stability_index(reference, current), 1.0)

    def test_constant_reference_detects_a_shift(self) -> None:
        reference = pd.Series([1.0] * 100)
        current = pd.Series([2.0] * 100)
        self.assertGreater(population_stability_index(reference, current), 1.0)

    def test_identical_categories_have_zero_divergence(self) -> None:
        values = pd.Series(["a", "a", "b", None])
        self.assertAlmostEqual(categorical_js_divergence(values, values), 0.0)

    def test_changed_categories_have_positive_divergence(self) -> None:
        reference = pd.Series(["a"] * 90 + ["b"] * 10)
        current = pd.Series(["a"] * 10 + ["b"] * 90)
        self.assertGreater(categorical_js_divergence(reference, current), 0.2)

    def test_measurement_labels_high_drift(self) -> None:
        train = pd.DataFrame({"numeric": range(100), "category": ["a"] * 100})
        test = pd.DataFrame({"numeric": range(200, 300), "category": ["b"] * 100})
        records = measure_drift(
            train,
            test,
            features=("numeric", "category"),
            categorical=("category",),
            fold=1,
        )
        self.assertEqual([record.level for record in records], ["high", "high"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from cohortshift.metrics import (
    FoldMetrics,
    aggregate_metrics,
    classification_metrics,
    expected_calibration_error,
)


class MetricsTests(unittest.TestCase):
    def test_perfect_confident_predictions_have_zero_calibration_error(self) -> None:
        truth = np.array(["a", "b", "a", "b"])
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(expected_calibration_error(truth, truth, probabilities), 0.0)

    def test_binary_metrics_include_auc(self) -> None:
        truth = np.array(["a", "b", "a", "b"])
        predictions = np.array(["a", "b", "a", "a"])
        probabilities = np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4]])
        metrics = classification_metrics(
            fold=1,
            train_rows=10,
            truth=truth,
            predictions=predictions,
            probabilities=probabilities,
            classes=np.array(["a", "b"]),
        )
        self.assertEqual(metrics.accuracy, 0.75)
        self.assertIsNotNone(metrics.roc_auc)

    def test_auc_is_none_when_test_class_is_missing(self) -> None:
        truth = np.array(["a", "a"])
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
        metrics = classification_metrics(
            fold=1,
            train_rows=10,
            truth=truth,
            predictions=truth,
            probabilities=probabilities,
            classes=np.array(["a", "b"]),
        )
        self.assertIsNone(metrics.roc_auc)

    def test_aggregate_reports_fold_count(self) -> None:
        fold = FoldMetrics(1, 10, 4, 0.5, 0.5, 0.5, 0.0, 0.7, 0.2, None)
        aggregate = aggregate_metrics([fold, fold])
        self.assertEqual(aggregate["accuracy"]["observed_folds"], 2)
        self.assertEqual(aggregate["accuracy"]["mean"], 0.5)


if __name__ == "__main__":
    unittest.main()

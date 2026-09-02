from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cohortshift.evaluate import EvaluationError, evaluate_csv, evaluate_frame
from tests.helpers import make_config, make_frame


class EvaluationTests(unittest.TestCase):
    def test_end_to_end_evaluation_writes_reproducible_artifacts(self) -> None:
        frame = make_frame()
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_frame(frame, make_config(), Path(directory, "report"))
            self.assertEqual(result.folds, 3)
            self.assertEqual(result.rows, len(frame))
            report = json.loads(Path(result.report).read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(len(report["fold_metrics"]), 3)
            self.assertEqual(report["source"]["rows"], len(frame))
            self.assertEqual(len(report["source"]["frame_sha256"]), 64)
            self.assertTrue(report["feature_importance"])
            predictions = pd.read_csv(result.predictions)
            self.assertEqual(set(predictions["fold"]), {1, 2, 3})

    def test_evaluation_is_deterministic(self) -> None:
        frame = make_frame()
        with tempfile.TemporaryDirectory() as directory:
            first = evaluate_frame(frame, make_config(), Path(directory, "first"))
            second = evaluate_frame(frame, make_config(), Path(directory, "second"))
            first_report = json.loads(Path(first.report).read_text(encoding="utf-8"))
            second_report = json.loads(Path(second.report).read_text(encoding="utf-8"))
            self.assertEqual(first_report["fold_metrics"], second_report["fold_metrics"])
            self.assertEqual(
                first_report["feature_importance"],
                second_report["feature_importance"],
            )

    def test_target_copy_blocks_evaluation(self) -> None:
        frame = make_frame()
        frame["leak"] = frame["outcome"]
        config = make_config(feature_columns=("signal", "leak"), categorical_columns=())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EvaluationError, "target-copy"):
                evaluate_frame(frame, config, Path(directory))

    def test_audit_override_allows_nonstructural_error(self) -> None:
        frame = make_frame()
        frame["leak"] = frame["outcome"]
        config = make_config(
            feature_columns=("signal", "leak"),
            categorical_columns=(),
            fail_on_audit_errors=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_frame(frame, config, Path(directory))
            self.assertEqual(result.folds, 3)

    def test_unseen_test_class_fails_explicitly(self) -> None:
        frame = make_frame()
        latest_year = pd.to_datetime(frame["sample_date"]).dt.year.max()
        frame.loc[pd.to_datetime(frame["sample_date"]).dt.year == latest_year, "outcome"] = 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EvaluationError, "unseen classes"):
                evaluate_frame(frame, make_config(folds=1), Path(directory))

    def test_csv_evaluation_records_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "data.csv")
            make_frame().to_csv(source, index=False)
            result = evaluate_csv(source, make_config(), Path(directory, "report"))
            report = json.loads(Path(result.report).read_text(encoding="utf-8"))
            self.assertEqual(report["source"]["name"], "data.csv")
            self.assertEqual(len(report["source"]["sha256"]), 64)

    @unittest.skipIf(os.name == "nt", "POSIX file modes are not available")
    def test_report_artifacts_are_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_frame(make_frame(), make_config(), Path(directory))
            self.assertEqual(Path(result.report).stat().st_mode & 0o777, 0o600)
            self.assertEqual(Path(result.metrics).stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()

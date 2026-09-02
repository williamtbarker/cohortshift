from __future__ import annotations

import unittest

import pandas as pd

from cohortshift.audit import audit_frame
from tests.helpers import make_config, make_frame


class AuditTests(unittest.TestCase):
    def test_clean_frame_has_no_error_findings(self) -> None:
        findings = audit_frame(make_frame(), make_config())
        self.assertFalse(any(finding.severity == "error" for finding in findings))

    def test_finds_missing_columns(self) -> None:
        frame = make_frame().drop(columns="signal")
        findings = audit_frame(frame, make_config())
        self.assertIn("missing-column", {finding.code for finding in findings})

    def test_finds_target_copy(self) -> None:
        frame = make_frame()
        frame["leak"] = frame["outcome"]
        config = make_config(feature_columns=("signal", "leak", "site"))
        findings = audit_frame(frame, config)
        errors = {(finding.code, finding.column) for finding in findings}
        self.assertIn(("target-copy", "leak"), errors)

    def test_finds_undeclared_categorical_feature(self) -> None:
        config = make_config(categorical_columns=())
        findings = audit_frame(make_frame(), config)
        self.assertIn("non-numeric-feature", {finding.code for finding in findings})

    def test_reports_missingness_constant_and_duplicates(self) -> None:
        frame = make_frame()
        frame["noise"] = 1.0
        frame.loc[:120, "signal"] = None
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        codes = {finding.code for finding in audit_frame(frame, make_config())}
        self.assertTrue({"constant-feature", "high-missingness", "duplicate-rows"} <= codes)


if __name__ == "__main__":
    unittest.main()

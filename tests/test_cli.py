from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from cohortshift.cli import main
from tests.helpers import make_frame


class CliTests(unittest.TestCase):
    def test_audit_plan_and_evaluate_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.csv"
            output = root / "report"
            make_frame().to_csv(source, index=False)
            common = [
                str(source),
                "--time",
                "sample_date",
                "--target",
                "outcome",
                "--features",
                "signal,noise,site",
                "--categorical",
                "site",
                "--group",
                "entity_id",
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["audit", *common]), 0)
                self.assertEqual(main(["plan", *common]), 0)
                self.assertEqual(
                    main(["evaluate", *common, "--output", str(output)]),
                    0,
                )
            self.assertTrue((output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()

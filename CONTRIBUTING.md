# Contributing

Issues and focused pull requests are welcome.

1. Install development dependencies with `python -m pip install -e '.[dev]'`.
2. Use synthetic data in tests and bug reports.
3. Add a regression test for changes to splitting, audits, metrics, or drift calculations.
4. Run `python -m ruff format .` and `python -m ruff check --fix .`.
5. Run `./scripts/verify.sh`.

Changes must preserve the core temporal contract: preprocessing and fitting use training rows
only, and no training timestamp may overlap or follow its fold's test window.

By contributing, you agree that your contribution may be distributed under the MIT License.

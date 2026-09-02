#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

python3 -m compileall -q src tests examples
PYTHONPATH=src python3 -m unittest discover -s tests -v

if python3 -c 'import ruff' 2>/dev/null; then
  python3 -m ruff format --check .
  python3 -m ruff check .
else
  echo "ruff not installed; skipping format and lint checks"
fi

if python3 -c 'import mypy' 2>/dev/null; then
  python3 -m mypy
else
  echo "mypy not installed; skipping static type checks"
fi

verify_dir="$(mktemp -d "${TMPDIR:-/tmp}/cohortshift-verify.XXXXXX")"
trap 'rm -rf "$verify_dir"' EXIT

python3 examples/generate_demo.py "$verify_dir/demo.csv" >/dev/null
common_args=(
  "$verify_dir/demo.csv"
  --time sample_date
  --target outcome
  --features signal,noise,site
  --categorical site
  --group entity_id
  --minimum-train-periods 3
)
PYTHONPATH=src python3 -m cohortshift.cli audit "${common_args[@]}" >/dev/null
PYTHONPATH=src python3 -m cohortshift.cli plan "${common_args[@]}" >/dev/null
PYTHONPATH=src python3 -m cohortshift.cli evaluate \
  "${common_args[@]}" --output "$verify_dir/report" >/dev/null
test -s "$verify_dir/report/report.json"
test "$(wc -l < "$verify_dir/report/fold_metrics.csv")" -eq 4

mkdir -p "$verify_dir/wheels"
if python3 -c 'import setuptools.build_meta' 2>/dev/null; then
  PIP_NO_INDEX=1 python3 -m pip wheel \
    --no-deps --no-build-isolation --wheel-dir "$verify_dir/wheels" .
else
  python3 -m pip wheel --no-deps --wheel-dir "$verify_dir/wheels" .
fi

echo "All checks passed, including temporal evaluation and wheel creation."

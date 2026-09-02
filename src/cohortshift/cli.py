"""Command-line interface for CohortShift."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from cohortshift import __version__
from cohortshift.audit import audit_frame
from cohortshift.config import EvaluationConfig
from cohortshift.evaluate import EvaluationError, evaluate_csv
from cohortshift.split import temporal_folds


def _columns(value: str) -> tuple[str, ...]:
    columns = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not columns:
        raise argparse.ArgumentTypeError("provide at least one column")
    return columns


def _add_configuration(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", type=Path)
    parser.add_argument("--time", required=True, dest="time_column")
    parser.add_argument("--target", required=True, dest="target_column")
    parser.add_argument("--features", required=True, type=_columns, dest="feature_columns")
    parser.add_argument("--categorical", type=_columns, default=(), dest="categorical_columns")
    parser.add_argument("--group", dest="group_column")
    parser.add_argument("--frequency", choices=("Y", "Q", "M", "W", "D"), default="Y")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-periods", type=int, default=1)
    parser.add_argument("--gap-periods", type=int, default=0)
    parser.add_argument("--minimum-train-periods", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=42)


def _config(namespace: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        time_column=namespace.time_column,
        target_column=namespace.target_column,
        feature_columns=namespace.feature_columns,
        categorical_columns=namespace.categorical_columns,
        group_column=namespace.group_column,
        frequency=namespace.frequency,
        folds=namespace.folds,
        test_periods=namespace.test_periods,
        gap_periods=namespace.gap_periods,
        minimum_train_periods=namespace.minimum_train_periods,
        random_seed=namespace.random_seed,
        fail_on_audit_errors=not getattr(namespace, "allow_audit_errors", False),
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path.expanduser().resolve())
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        raise EvaluationError(f"cannot read CSV: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cohortshift",
        description="Audit and evaluate classifiers with time-respecting folds.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="run schema and leakage checks")
    _add_configuration(audit)

    plan = subparsers.add_parser("plan", help="show temporal folds without fitting a model")
    _add_configuration(plan)

    evaluate = subparsers.add_parser("evaluate", help="fit and evaluate expanding-window folds")
    _add_configuration(evaluate)
    evaluate.add_argument("--output", type=Path, default=Path("cohortshift-report"))
    evaluate.add_argument(
        "--allow-audit-errors",
        action="store_true",
        help="continue despite error-level audit findings",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(arguments)
    try:
        config = _config(namespace)
        if namespace.command == "audit":
            findings = audit_frame(_read_csv(namespace.csv), config)
            print(json.dumps([finding.as_dict() for finding in findings], indent=2))
            return 1 if any(finding.severity == "error" for finding in findings) else 0
        if namespace.command == "plan":
            folds = temporal_folds(_read_csv(namespace.csv), config)
            print(json.dumps([fold.as_dict() for fold in folds], indent=2))
            return 0
        if namespace.command == "evaluate":
            result = evaluate_csv(namespace.csv, config, namespace.output)
            print(json.dumps(asdict(result), indent=2, sort_keys=True))
            return 0
        parser.error(f"unsupported command: {namespace.command}")
    except (EvaluationError, OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"cohortshift: error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

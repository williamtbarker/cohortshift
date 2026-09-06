# CohortShift

[![License](https://img.shields.io/github/license/williamtbarker/cohortshift)](https://github.com/williamtbarker/cohortshift/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/williamtbarker/cohortshift?display_name=tag&sort=semver)](https://github.com/williamtbarker/cohortshift/releases)

[![CI](https://github.com/williamtbarker/cohortshift/actions/workflows/ci.yml/badge.svg)](https://github.com/williamtbarker/cohortshift/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CohortShift is a Python CLI and library for leakage-resistant temporal evaluation of tabular
classifiers. It replaces optimistic random splits with expanding-window backtests, optionally
purges repeated entities across train and test, measures feature drift, and writes reproducible
audit artifacts.

It is aimed at datasets whose collection process changes over time: scientific observations,
manufacturing records, customer cohorts, surveillance data, and other longitudinal tables.

## The problem

A random train/test split asks whether a model can interpolate among records drawn from the same
mixture. Many deployed models face a harder question: can a model trained on the past work on a
future cohort whose prevalence, categories, instruments, or operating conditions have shifted?

CohortShift makes the time boundary explicit and records the evidence needed to inspect it:

- chronological expanding-window folds;
- configurable test windows and temporal embargo gaps;
- repeated-entity detection and train-side purging;
- preprocessing fitted separately inside each training fold;
- schema, missingness, target-copy, constant-feature, and cardinality audits;
- accuracy, balanced accuracy, macro F1, MCC, log loss, ROC AUC, and calibration error;
- numeric population stability index (PSI);
- categorical Jensen–Shannon divergence;
- per-fold predictions and model coefficients; and
- source, frame, configuration, and software-version provenance.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python examples/generate_demo.py demo.csv

cohortshift audit demo.csv \
  --time sample_date \
  --target outcome \
  --features signal,noise,site \
  --categorical site \
  --group entity_id

cohortshift plan demo.csv \
  --time sample_date \
  --target outcome \
  --features signal,noise,site \
  --categorical site \
  --group entity_id

cohortshift evaluate demo.csv \
  --time sample_date \
  --target outcome \
  --features signal,noise,site \
  --categorical site \
  --group entity_id \
  --folds 3 \
  --minimum-train-periods 3 \
  --output demo-report
```

The example is deterministic synthetic data and is not derived from people, patients, or private
research.

## Temporal contract

For every fold, all training periods precede the optional gap and every test period:

```text
past training periods | embargo gap | test periods | unused future periods
```

With `--group`, any entity appearing in the test partition is removed from that fold's training
partition. This measures generalization to unseen entities as well as future time. The report
records the number of overlapping groups and purged rows; if purging empties a partition, the
evaluation fails explicitly.

The latest requested windows become test folds. Earlier periods form the initial training
history, which grows with each fold.

| Option | Default | Meaning |
|---|---:|---|
| `--frequency` | `Y` | Calendar period: `Y`, `Q`, `M`, `W`, or `D` |
| `--folds` | 3 | Number of chronological test folds |
| `--test-periods` | 1 | Consecutive periods in each test fold |
| `--gap-periods` | 0 | Embargoed periods between train and test |
| `--minimum-train-periods` | 2 | Smallest permissible training history |
| `--random-seed` | 42 | Estimator seed recorded in the report |

## Audit behavior

`cohortshift audit` reports findings as JSON. Error-level findings include missing required
columns, invalid times, missing targets, undeclared nonnumeric features, and features that exactly
copy the target. Warnings include high missingness, constants, duplicate rows, extremely
high-cardinality categories, and near-perfect numeric target correlation.

Evaluation stops on errors by default. `--allow-audit-errors` can override a suspected target-copy
finding, but it cannot bypass structurally impossible input such as missing columns or invalid
times.

## Modeling behavior

Version 0.1 deliberately uses one transparent baseline:

- median imputation and standardization for numeric features;
- most-frequent imputation and one-hot encoding for categorical features; and
- class-balanced logistic regression.

The complete preprocessing pipeline is fitted independently within each training fold. No test
statistics are used for imputation, scaling, encoding, or fitting. Mean absolute coefficients are
reported as diagnostic model signals, not causal effects or universally comparable importance
scores.

## Drift interpretation

Numeric PSI uses quantile bins learned from the training partition and includes missing values as
a separate bin. Categorical drift uses Jensen–Shannon divergence over the union of train and test
categories.

| Measure | Stable | Moderate | High |
|---|---:|---:|---:|
| Numeric PSI | `< 0.10` | `0.10–0.25` | `≥ 0.25` |
| Categorical JS divergence | `< 0.05` | `0.05–0.15` | `≥ 0.15` |

These thresholds are triage conventions, not universal decision rules.

## Output artifacts

```text
report.json         configuration, provenance, folds, metrics, drift summary, coefficients
audit.csv           schema and leakage findings
fold_metrics.csv    one row per temporal fold
drift.csv           one row per feature and fold
predictions.csv     row position, time, actual/predicted class, and probabilities
```

Files are written through temporary files and atomically replaced. On POSIX systems they receive
owner-only permissions because predictions and source-derived summaries may be sensitive.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff format .
python -m ruff check --fix .
./scripts/verify.sh
```

The verifier compiles the package, runs 32 unit and integration tests, checks formatting and
typing, evaluates a synthetic longitudinal dataset, and builds a wheel. CI runs the complete suite
on Python 3.10 through 3.13.

## Scope and limitations

CohortShift evaluates tabular classification and does not train neural networks, select production
thresholds, establish causality, or prove deployment safety. PSI and JS divergence summarize
marginal feature movement, not joint or conditional drift. The default classifier is a diagnostic
baseline; domain-specific models should be compared under the same fold contract.

Do not use the output as a clinical, regulatory, or automated deployment decision without an
appropriate domain-specific validation process.

## License

MIT. See [LICENSE](LICENSE).

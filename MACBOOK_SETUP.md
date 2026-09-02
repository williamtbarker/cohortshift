# macOS setup and GitHub release

These commands assume the repository is `~/Documents/cohortshift`.

## Verify locally

```bash
cd ~/Documents/cohortshift
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m ruff format .
python -m ruff check --fix .
./scripts/verify.sh
```

The initial Ruff commands may change formatting and import order. The verifier itself does not
modify source files.

## Run the demonstration

```bash
python examples/generate_demo.py demo.csv

cohortshift evaluate demo.csv \
  --time sample_date \
  --target outcome \
  --features signal,noise,site \
  --categorical site \
  --group entity_id \
  --minimum-train-periods 3 \
  --output demo-report
```

Both `demo.csv` and `demo-report/` are ignored by Git.

## Publish

```bash
git init
git add .
git commit -m "Initial release: leakage-resistant temporal model evaluation"
git branch -M main
gh repo create cohortshift --public --source=. --remote=origin --push
```

Recommended topics:

```text
python machine-learning temporal-validation data-drift model-evaluation reproducibility
```

## Optional tagged release

```bash
git tag -a v0.1.0 -m "CohortShift v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --generate-notes
```

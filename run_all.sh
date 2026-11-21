#!/usr/bin/env bash
# Re-run the whole pipeline (data download, EDA, preprocessing, models, evaluation) and time it.
# Usage: ./run_all.sh   (expects the venv at .venv; see README for setup)
set -euo pipefail
cd "$(dirname "$0")/notebooks"
start=$(date +%s)
for nb in 01_eda 02_preprocess 03_models 04_eval; do
  t0=$(date +%s)
  ../.venv/bin/jupytext -q --update --to ipynb "$nb.py"
  ../.venv/bin/jupyter nbconvert --log-level=ERROR --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 "$nb.ipynb"
  echo "$nb: $(( $(date +%s) - t0 )) s"
done
echo "Total: $(( $(date +%s) - start )) s"

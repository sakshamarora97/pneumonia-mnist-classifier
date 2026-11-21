# Pneumonia vs. normal chest X-ray classification (PneumoniaMNIST)

A minimal, reproducible pipeline that classifies 28×28 pediatric chest X-rays as normal or pneumonia. It is meant as a triage aid that supports clinicians, not a diagnostic tool.

## Layout
| Path | Purpose |
| --- | --- |
| `config.py` | Seed, paths, image size, `set_seeds()` |
| `src/data.py` | `load_splits()`: TFDS download, cached to `data/pneumonia_28.npz` |
| `src/metrics.py` | `evaluate()` and `log_result()` → `results/val_metrics.csv` |
| `notebooks/01_eda` | Tasks 2–7: splits, class balance, figures 1–6, duplicates, brightness shift |
| `notebooks/02_preprocess` | Task 8: scaling, flattening, standardisation, class weights |
| `notebooks/03_models` | Tasks 9–12: majority baseline, logistic regression, PCA + random forest, CNN |
| `notebooks/04_eval` | Tasks 14–17: thresholds on validation, single test run with bootstrap CIs, error analysis |
| `run_all.sh` | Rebuilds every result from scratch and times each notebook |
| `figures/` | Report figures |
| `results/` | Validation metrics table and saved CNN |

Notebooks are kept as paired `.py` (jupytext percent format, easy to diff) and `.ipynb` files.

## Setup and run
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./run_all.sh   # downloads the data and re-runs notebooks 01–04, printing the time for each
```

## Dataset facts (verified in `01_eda`)
| Split | Images | Normal (0) | Pneumonia (1) | % pneumonia |
| --- | --- | --- | --- | --- |
| train | 4,708 | 1,214 | 3,494 | 74.2 |
| val | 524 | 135 | 389 | 74.2 |
| test | 624 | 234 | 390 | 62.5 |

- Images are 28×28×1 uint8. TFDS label names are `['Normal', 'Pneumonia']`, so 1 = pneumonia.
- Exact duplicates: 18 within train, 3 within test, and 8 images shared between train and validation (a small leak). None are shared with test.
- Test differs from train in class balance and brightness, so expect lower test scores than validation scores.

## Validation results (threshold 0.5)
| Model | AUC | Sensitivity | Specificity | Train time |
| --- | --- | --- | --- | --- |
| It0 majority | 0.500 | 1.000 | 0.000 | 0 s |
| It1 logistic regression (C=0.01) | 0.992 | 0.951 | 0.963 | 0.2 s |
| It2 PCA(50) + random forest | 0.987 | 0.972 | 0.926 | 1.4 s |
| It3 CNN | 0.992 | 0.774 | 0.993 | 196 s |

The CNN's low sensitivity at 0.5 is a threshold issue, not a ranking issue.

## Test results (`04_eval`, thresholds chosen on validation for ≥95% sensitivity)
| Model | Threshold | AUC (95% CI) | Sensitivity (95% CI) | Specificity (95% CI) | Missed pneumonia | False alarms |
| --- | --- | --- | --- | --- | --- | --- |
| It1 logistic regression | 0.50 | 0.930 (0.904–0.954) | 0.972 (0.955–0.987) | 0.709 (0.647–0.767) | 11 | 68 |
| It2 PCA + random forest | 0.55 | 0.941 (0.922–0.960) | 0.936 (0.910–0.959) | 0.774 (0.718–0.827) | 25 | 53 |
| **It3 CNN** | 0.06 | **0.961 (0.946–0.974)** | 0.969 (0.951–0.985) | 0.722 (0.662–0.780) | 12 | 65 |

The test set was scored once. The CNN is the final model: it holds up best under the train/test shift. Its main weakness is specificity, at about 72%.

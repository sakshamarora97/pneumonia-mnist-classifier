"""Shared evaluation helpers. Positive class = 1 (pneumonia)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

from config import RESULTS_DIR

RESULTS_CSV = RESULTS_DIR / "val_metrics.csv"


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """Metrics from probability-like scores for the positive class."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def threshold_for_sensitivity(y_true, scores, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity is >= target (so specificity is as high as possible)."""
    pos = np.sort(np.asarray(scores)[np.asarray(y_true) == 1])
    # Allow at most floor((1 - target) * n_pos) positives below the threshold.
    k = int(np.floor((1 - target) * len(pos) + 1e-9))
    return float(pos[k])


def bootstrap_ci(y_true, scores, threshold: float, n_boot: int = 1000, seed: int = 42) -> dict:
    """Percentile 95% CIs for AUC, sensitivity and specificity by resampling cases with replacement."""
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    stats = {"auc": [], "sensitivity": [], "specificity": []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, s = y_true[idx], scores[idx]
        if yt.min() == yt.max():
            continue
        m = evaluate(yt, s, threshold)
        for k in stats:
            stats[k].append(m[k])
    return {f"{k}_ci": (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}


def log_result(name: str, metrics: dict, train_seconds: float) -> pd.DataFrame:
    """Append (or replace) one model's validation row and return the full table."""
    row = {"model": name, **metrics, "train_seconds": round(train_seconds, 1)}
    df = pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists() else pd.DataFrame()
    if not df.empty:
        df = df[df["model"] != name]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RESULTS_CSV, index=False)
    return df

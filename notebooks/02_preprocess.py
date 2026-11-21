# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Preprocessing (task 8)
# Scale to [0, 1], flatten + standardise for scikit-learn (scaler fit on train only), compute class weights.
# The given train/val/test splits are kept as-is.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import json

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_DIR, set_seeds
from src.data import load_splits

set_seeds()
data, _ = load_splits()

# %%
out = {}
for s, (X, y) in data.items():
    out[f"img_{s}"] = X.astype(np.float32) / 255.0          # [N, 28, 28, 1] for the CNN
    out[f"y_{s}"] = y

scaler = StandardScaler().fit(out["img_train"].reshape(len(out["img_train"]), -1))
for s in data:
    flat = out[f"img_{s}"].reshape(len(out[f"img_{s}"]), -1)
    out[f"flat_{s}"] = scaler.transform(flat).astype(np.float32)  # [N, 784] for sklearn

for k, v in out.items():
    print(f"{k:12s} {str(v.shape):18s} {v.dtype}  min={v.min():.2f} max={v.max():.2f}")

# %%
y_tr = out["y_train"]
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weights = {0: float(weights[0]), 1: float(weights[1])}
print("Balanced class weights:", class_weights)

np.savez_compressed(DATA_DIR / "processed.npz", **out)
(DATA_DIR / "class_weights.json").write_text(json.dumps(class_weights, indent=2))
print("Saved", DATA_DIR / "processed.npz")

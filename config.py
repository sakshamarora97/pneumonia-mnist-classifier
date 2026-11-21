"""Shared settings for the PneumoniaMNIST project."""
import os
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

SEED = 42
IMG_SIZE = 28
CLASS_NAMES = {0: "normal", 1: "pneumonia"}  # confirmed in 01_eda

for d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
    d.mkdir(exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and (if imported) TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

"""Load PneumoniaMNIST (28x28) from TensorFlow Datasets, cached as a small .npz."""
import numpy as np

from config import DATA_DIR

CACHE = DATA_DIR / "pneumonia_28.npz"
SPLITS = ("train", "val", "test")


def load_splits():
    """Return ({split: (X uint8 [N,28,28,1], y int64 [N])}, label_names)."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        data = {s: (z[f"X_{s}"], z[f"y_{s}"]) for s in SPLITS}
        return data, list(z["label_names"])

    import tensorflow_datasets as tfds

    data = {}
    info = None
    for s in SPLITS:
        (X, y), info = tfds.load(
            "pneumonia_mnist",
            split=s,
            data_dir=str(DATA_DIR / "tfds"),
            batch_size=-1,
            as_supervised=True,
            with_info=True,
        )
        data[s] = (np.asarray(X, dtype=np.uint8), np.asarray(y, dtype=np.int64))

    label_names = list(info.features["label"].names)
    np.savez_compressed(
        CACHE,
        label_names=np.array(label_names),
        **{f"X_{s}": data[s][0] for s in SPLITS},
        **{f"y_{s}": data[s][1] for s in SPLITS},
    )
    return data, label_names

import pickle
from pathlib import Path

import numpy as np

DATA = Path("data")


def load():
    X_tr = np.load(DATA / "X_train_raw.npy")
    X_te = np.load(DATA / "X_test_raw.npy")
    y_tr = np.load(DATA / "y_train.npy")
    y_te = np.load(DATA / "y_test.npy")
    names = list(np.load(DATA / "feature_names.npy", allow_pickle=True))
    with open(DATA / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return X_tr, X_te, y_tr, y_te, scaler, names


def classes():
    return [str(c) for c in np.load(DATA / "classes.npy", allow_pickle=True)]


def split_val(X, y, frac=0.2):
    """The last `frac` of the training jets, exactly what Keras's validation_split=0.2 held out in v1."""
    n = int(len(X) * (1 - frac))
    return X[:n], y[:n], X[n:], y[n:]
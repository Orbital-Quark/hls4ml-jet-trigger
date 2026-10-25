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
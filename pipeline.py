from types import SimpleNamespace

import numpy as np

from common import DATA, load, split_val
from drift import augment
from features import make_prep

TEACHERS = DATA / "teachers"
V1 = {}
V2 = dict(dedup=True, log=True, augment=True, distill=True)


def teacher_key(flags):
    return "-".join(f for f in ("dedup", "log", "augment") if flags.get(f)) or "plain"


def build_data(flags, with_targets=True):
    """Everything a model needs for one version: preprocessing fitted on the training jets, the (possibly
    drift-augmented) training set, the clean validation set, the raw test set, and distillation targets."""
    from train import soft_targets

    X_tr_raw, X_te_raw, y_tr, y_te, scaler, names = load()
    Xa_raw, ya, Xv_raw, yv = split_val(X_tr_raw, y_tr)
    prep = make_prep(flags, Xa_raw, scaler, names)
    std = X_tr_raw.std(axis=0)  # same drift size for every version
    X_fit_raw = augment(Xa_raw, std, np.random.default_rng(7), names) if flags.get("augment") else Xa_raw

    d = SimpleNamespace(names=names, prep=prep, std=std, X_fit=prep.transform(X_fit_raw), y_fit=ya,
                        X_val=prep.transform(Xv_raw), y_val=yv, X_val_raw=Xv_raw,
                        X_te_raw=X_te_raw, y_te=y_te, targets=None)
    if flags.get("distill") and with_targets:
        d.targets = soft_targets(ya, np.load(TEACHERS / f"{teacher_key(flags)}.npy"))
    return d


def ensure_teacher(flags):
    """Train the float, full-size teacher for these flags once and save its answers on the training jets."""
    path = TEACHERS / f"{teacher_key(flags)}.npy"
    if path.exists():
        return f"teacher {path.name} exists"
    from train import train

    d = build_data(flags, with_targets=False)
    model = train(dict(bits=None, scale=1.0, sparsity=0.0), d.X_fit, d.y_fit, d.X_val, d.y_val)
    TEACHERS.mkdir(parents=True, exist_ok=True)
    np.save(path, model.predict(d.X_fit, batch_size=8192, verbose=0).astype(np.float32))
    acc = float((model.predict(d.X_val, batch_size=8192, verbose=0).argmax(1) == d.y_val).mean())
    return f"teacher {path.name} trained, val acc {acc:.4f}"

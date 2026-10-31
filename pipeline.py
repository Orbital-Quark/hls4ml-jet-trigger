import json
from types import SimpleNamespace

import numpy as np

from common import DATA, classes, load, split_val
from drift import augment
from features import FeatureSelector, SelectedPrep, make_prep
from weights import flat_mass_weights, rate_weights

TEACHERS = DATA / "teachers"
SELECTIONS = DATA / "selections"
V1 = {}
V2 = dict(dedup=True, log=True, augment=True, distill=True)
V3 = dict(V2, rate=True, flat_mass=True, select=True)
# v4: v2, plus the one v3 change that helped (rate weights: W/Z trigger efficiency), plus the self-correcting trigger
V4 = dict(V2, rate=True, adaptive=True)


def teacher_key(flags):
    return "-".join(f for f in ("dedup", "log", "augment") if flags.get(f)) or "plain"


def selection_key(flags):
    return "-".join(f for f in ("dedup", "log", "augment", "rate", "flat_mass") if flags.get(f)) or "plain"


def build_data(flags, with_targets=True):
    """Everything a model needs for one version: preprocessing fitted on the training jets, the (possibly
    drift-augmented) training set, the clean validation set, the raw test set, distillation targets
    and per-jet training weights."""
    from train import soft_targets

    X_tr_raw, X_te_raw, y_tr, y_te, scaler, names = load()
    Xa_raw, ya, Xv_raw, yv = split_val(X_tr_raw, y_tr)
    prep = make_prep(flags, Xa_raw, scaler, names)
    if flags.get("select"):
        prep = SelectedPrep(prep, json.loads((SELECTIONS / f"{selection_key(flags)}.json").read_text())["keep"])
    std = X_tr_raw.std(axis=0)  # same drift size for every version
    X_fit_raw = augment(Xa_raw, std, np.random.default_rng(7), names) if flags.get("augment") else Xa_raw

    d = SimpleNamespace(names=names, prep=prep, std=std, X_fit=prep.transform(X_fit_raw), y_fit=ya,
                        X_val=prep.transform(Xv_raw), y_val=yv, X_val_raw=Xv_raw,
                        X_te_raw=X_te_raw, y_te=y_te, targets=None, weights=None)
    if flags.get("distill") and with_targets:
        d.targets = soft_targets(ya, np.load(TEACHERS / f"{teacher_key(flags)}.npy"))
    if flags.get("rate") or flags.get("flat_mass"):
        cl = classes()
        d.weights = np.ones(len(ya), dtype=np.float32)
        if flags.get("rate"):
            d.weights *= rate_weights(ya, cl)
        if flags.get("flat_mass"):
            d.weights *= flat_mass_weights(X_fit_raw[:, names.index("mass_mmdt")], ya, len(cl))
    return d


def ensure_teacher(flags):
    """Train the float, full-size teacher for these flags once and save its answers on the training jets."""
    path = TEACHERS / f"{teacher_key(flags)}.npy"
    if path.exists():
        return f"teacher {path.name} exists"
    from train import train

    d = build_data(dict(flags, select=False), with_targets=False)  # the teacher sees every input, unweighted
    model = train(dict(bits=None, scale=1.0, sparsity=0.0), d.X_fit, d.y_fit, d.X_val, d.y_val)
    TEACHERS.mkdir(parents=True, exist_ok=True)
    np.save(path, model.predict(d.X_fit, batch_size=8192, verbose=0).astype(np.float32))
    acc = float((model.predict(d.X_val, batch_size=8192, verbose=0).argmax(1) == d.y_val).mean())
    return f"teacher {path.name} trained, val acc {acc:.4f}"


def ensure_selection(flags):
    """Run FeatureSelector once for these flags, on a float full-size model trained the same way
    (same preprocessing and weights), and save which inputs to keep."""
    path = SELECTIONS / f"{selection_key(flags)}.json"
    if path.exists():
        return f"selection {path.name} exists: {json.loads(path.read_text())['keep']}"
    from train import train

    d = build_data(dict(flags, select=False, distill=False), with_targets=False)
    model = train(dict(bits=None, scale=1.0, sparsity=0.0), d.X_fit, d.y_fit, d.X_val, d.y_val, weights=d.weights)
    sel = FeatureSelector().fit(model, d.X_val, d.y_val, d.prep.names)
    SELECTIONS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(keep=sel.names, dropped_in_order=sel.steps, base_val_acc=sel.base), indent=1))
    return f"selection {path.name}: kept {len(sel.names)}, dropped (name, total loss) {sel.steps}"

import numpy as np

DRIFTS = (
    [("none", 0.0)]
    + [("mass_scale", d) for d in (0.02, 0.05, 0.10)]   # calorimeter calibration drift
    + [("smear", k) for k in (0.05, 0.10, 0.20)]        # resolution loss (ageing)
    + [("pileup", lam) for lam in (2, 5, 10)]            # extra soft particles (pileup proxy)
)


def apply_drift(X_raw, kind, level, train_std, rng, names):
    X = X_raw.astype(np.float32).copy()
    mass, mult = names.index("mass_mmdt"), names.index("multiplicity")
    if kind == "mass_scale":
        X[:, mass] *= 1.0 + level
    elif kind == "smear":
        noise = rng.normal(0.0, 1.0, X.shape).astype(np.float32) * (level * train_std)
        noise[:, mult] = 0.0  # multiplicity is a count; handled by "pileup"
        X += noise
    elif kind == "pileup":
        X[:, mult] += rng.poisson(level, size=len(X))
    return X


def augment(X_raw, train_std, rng, names, frac=0.5):
    """Drift-aware training data (v2): a random `frac` of jets each get one random drift at a random level.
    Levels are drawn from ranges that contain the test levels (mass in both directions)."""
    X = X_raw.astype(np.float32).copy()
    mass, mult = names.index("mass_mmdt"), names.index("multiplicity")
    kind = np.where(rng.random(len(X)) < frac, rng.integers(1, 4, len(X)), 0)  # 0 clean, 1 mass, 2 smear, 3 pileup

    m = kind == 1
    X[m, mass] *= 1.0 + rng.uniform(-0.10, 0.10, m.sum()).astype(np.float32)
    s = kind == 2
    noise = rng.normal(0.0, 1.0, (s.sum(), X.shape[1])).astype(np.float32)
    noise *= rng.uniform(0.0, 0.20, (s.sum(), 1)).astype(np.float32) * train_std
    noise[:, mult] = 0.0
    X[s] += noise
    p = kind == 3
    X[p, mult] += rng.poisson(rng.uniform(0.0, 10.0, p.sum()))
    return X
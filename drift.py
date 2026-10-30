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
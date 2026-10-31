import numpy as np

# Rough share of each jet type among high-energy (~1 TeV) jets at the LHC, relative to one signal type.
# Ordinary quark and gluon jets are produced roughly 1,000 times more often than boosted W, Z or top jets.
# This is an order-of-magnitude assumption (stated in the report), not a measured cross-section.
PRODUCTION_RATE = {"g": 1000.0, "q": 1000.0, "t": 1.0, "w": 1.0, "z": 1.0}
SOFTEN = 0.25      # weight = (real share / dataset share) ** SOFTEN; at 1 the rare types would barely count
MASS_BINS = 40
FLATTEN = 0.5      # 1 = every class fully flat in mass; 0.5 = halfway, so mass still helps but can't dominate
MAX_WEIGHT = 10.0  # cap (before renormalising) for jets in mass bins where their class is very rare


def rate_weights(y, classes):
    """Custom (v3): per-jet weights that move the balanced dataset towards the real-world mix of jet types.
    The dataset's own class shares are read from the labels, so the weights undo whatever mix the data has.
    Mistakes on the common q/g jets then cost more, as they do in a trigger."""
    share = np.bincount(y, minlength=len(classes)) / len(y)
    real = np.array([PRODUCTION_RATE[c] for c in classes])
    w = ((real / real.sum()) / share) ** SOFTEN
    w = w[y]
    return (w / w.mean()).astype(np.float32)


def flat_mass_weights(mass, y, n_classes):
    """Custom (v3): reweight jets so every class has (partly) the same, flat jet-mass distribution.
    Mass bins are quantiles of the training jets; a jet is weighted up where its class is rare at its mass.
    Each class keeps its total weight, so only the mass shape changes and the model has to rely more on
    the jet's prong structure than on mass, which drifts with the detector calibration."""
    edges = np.quantile(mass, np.linspace(0, 1, MASS_BINS + 1))
    b = np.clip(np.searchsorted(edges, mass, side="right") - 1, 0, MASS_BINS - 1)
    counts = np.zeros((n_classes, MASS_BINS))
    np.add.at(counts, (y, b), 1)
    density = counts[y, b] / counts.sum(axis=1)[y] * MASS_BINS  # 1 where the class is already flat
    w = np.minimum(density ** -FLATTEN, MAX_WEIGHT)
    for c in range(n_classes):
        w[y == c] /= w[y == c].mean()
    return w.astype(np.float32)

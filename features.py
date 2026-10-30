import numpy as np


class PhysicsScaler:
    """Data-driven replacement for sklearn's StandardScaler (v2).

    Every choice is read off the training jets, not hard-coded:
      - features that are exact copies of an earlier feature are dropped
      - features spanning more than LOG_SPAN between their 1% and 99% quantiles are log-transformed
      - with log on, each feature is centred on its median and scaled by its interquartile range (robust to tails)
    On this dataset that drops d2_a1_b1_mmdt and logs c1_b1, c1_b2, c2_b1, c2_b2 and mass_mmdt.
    """

    LOG_SPAN = 100.0

    def __init__(self, dedup=True, log=True):
        self.dedup, self.log = dedup, log

    def fit(self, X, names):
        keep = []
        for j in range(X.shape[1]):
            if not (self.dedup and any(np.array_equal(X[:, i], X[:, j]) for i in keep)):
                keep.append(j)
        self.keep = keep
        self.names = [str(names[j]) for j in keep]
        self.dropped = [str(names[j]) for j in range(X.shape[1]) if j not in keep]

        Xk = X[:, keep].astype(np.float64)
        lo, hi = np.quantile(Xk, [0.01, 0.99], axis=0)
        self.floor = Xk.min(axis=0)
        self.log_mask = self.log & (self.floor > 0) & (hi > self.LOG_SPAN * np.where(lo > 0, lo, np.inf))

        Z = self._log(Xk)
        if self.log:  # robust centring and scaling, from the training quantiles
            q25, self.center, q75 = np.quantile(Z, [0.25, 0.5, 0.75], axis=0)
            self.scale = q75 - q25
        else:         # dedup only: plain mean/std, so the ablation isolates dropping the duplicate
            self.center, self.scale = Z.mean(axis=0), Z.std(axis=0)
        return self

    def _log(self, Xk):
        Z = Xk.copy()
        # drifted (smeared) values can fall below anything seen in training; clip them to the training minimum
        Z[:, self.log_mask] = np.log(np.maximum(Xk[:, self.log_mask], self.floor[self.log_mask]))
        return Z

    def transform(self, X):
        Z = self._log(X[:, self.keep].astype(np.float64))
        return ((Z - self.center) / self.scale).astype(np.float32)

    def describe(self):
        logged = [n for n, m in zip(self.names, self.log_mask) if m]
        return f"dropped {self.dropped}, log-transformed {logged}, {len(self.names)} inputs"


class StandardPrep:
    """v1: the saved StandardScaler on all 16 features, exactly as in 5.1."""

    def __init__(self, scaler, names):
        self.scaler, self.names, self.dropped = scaler, list(names), []

    def transform(self, X):
        return self.scaler.transform(X).astype(np.float32)

    def describe(self):
        return f"StandardScaler, {len(self.names)} inputs"


def make_prep(flags, X_tr_raw, scaler, names):
    if flags.get("dedup") or flags.get("log"):
        return PhysicsScaler(dedup=flags.get("dedup", False), log=flags.get("log", False)).fit(X_tr_raw, names)
    return StandardPrep(scaler, names)

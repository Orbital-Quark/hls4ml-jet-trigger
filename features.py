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


class FeatureSelector:
    """Custom (v3): drop the inputs a trained model barely uses, measured on validation jets.

    Shuffling a feature across jets breaks its link to the answer; the accuracy lost is how much the model
    needs it. Features are dropped one at a time, each time the one whose shuffling hurts least, on top of
    the ones already dropped (still shuffled). That matters because two overlapping features each look
    useless while the other is present. Stops before the total loss would exceed MAX_LOSS.
    """

    MAX_LOSS = 0.003  # 0.3 accuracy points in total, a few times the run-to-run noise

    def fit(self, model, X_val, y_val, names, seed=0):
        perm = np.random.default_rng(seed).permutation(len(X_val))  # one fixed shuffle for every comparison

        def acc(X):
            return float((model.predict(X, batch_size=8192, verbose=0).argmax(1) == y_val).mean())

        self.base = acc(X_val)
        Xs, dropped, self.steps = X_val.copy(), [], []
        while len(dropped) < X_val.shape[1] - 1:
            trials = {}
            for j in (j for j in range(X_val.shape[1]) if j not in dropped):
                Xt = Xs.copy()
                Xt[:, j] = X_val[perm, j]
                trials[j] = acc(Xt)
            j = max(trials, key=trials.get)
            if self.base - trials[j] > self.MAX_LOSS:
                break
            dropped.append(j)
            Xs[:, j] = X_val[perm, j]
            self.steps.append((str(names[j]), round(self.base - trials[j], 5)))
        self.names = [str(n) for j, n in enumerate(names) if j not in dropped]
        self.dropped = [s[0] for s in self.steps]
        return self


class SelectedPrep:
    """Any preprocessing, followed by keeping only the selected inputs (v3)."""

    def __init__(self, prep, keep):
        self.prep, self.names = prep, list(keep)
        self.cols = [prep.names.index(n) for n in keep]
        self.dropped = prep.dropped + [n for n in prep.names if n not in keep]

    def transform(self, X):
        return np.ascontiguousarray(self.prep.transform(X)[:, self.cols])

    def describe(self):
        return f"{self.prep.describe()}; feature selection keeps {len(self.names)}: {self.names}"


def make_prep(flags, X_tr_raw, scaler, names):
    if flags.get("dedup") or flags.get("log"):
        return PhysicsScaler(dedup=flags.get("dedup", False), log=flags.get("log", False)).fit(X_tr_raw, names)
    return StandardPrep(scaler, names)

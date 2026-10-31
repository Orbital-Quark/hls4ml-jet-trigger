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


class StandardPrep:
    """v1: the saved StandardScaler on all 16 features, exactly as in 5.1."""

    def __init__(self, scaler, names):
        self.scaler, self.names, self.dropped = scaler, list(names), []

    def transform(self, X):
        return self.scaler.transform(X).astype(np.float32)



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



class PaddedPrep:
    """Any preprocessing, followed by always-zero inputs up to `n` (v5). hls4ml only accepts reuse factors that
    divide a layer evenly: 15 inputs rule out reuse 4, so it takes 5 and the chip accepts a new jet every 25 ns
    instead of 20. A 16th input that is always zero changes nothing in the answer and brings reuse 4 back."""

    def __init__(self, prep, n):
        self.prep, self.n = prep, n
        self.names = prep.names + [f"pad{i}" for i in range(n - len(prep.names))]
        self.dropped = prep.dropped

    def transform(self, X):
        Z = self.prep.transform(X)
        return np.ascontiguousarray(np.pad(Z, ((0, 0), (0, self.n - Z.shape[1]))))



CAL_BLOCK = 256   # jets per update, as for the self-correcting trigger: a small counter on the chip
CAL_GAIN = 0.02   # how far one update can move an estimate, as a fraction of it


class InputCalibrator:
    """v5.2: physics-based self-calibration of the two inputs a drifting detector shifts.

    A trigger's stream is dominated by ordinary quark/gluon jets (~1000 per W, Z or top jet), whose jet mass and
    particle count are stable physics. Their medians are a standard candle: if the median moves, the detector
    moved. The medians on the healthy detector are stored once. On the live stream, a running estimate of each
    median is nudged every CAL_BLOCK jets towards the side with more jets (a counter and a register on the chip),
    and every jet is corrected with the current estimate:
      mass           x (healthy median / current median)     undoes an energy-scale slip
      particle count - (current median - healthy median)     undoes extra pileup particles
    Smearing is noise around the true value, so it leaves the medians where they are and is left alone."""

    def __init__(self, names):
        self.cols = [names.index("mass_mmdt"), names.index("multiplicity")]

    def fit(self, X_raw):
        self.ref = np.median(X_raw[:, self.cols], axis=0).astype(np.float64)
        return self

    def run(self, X_warm, X_raw):
        """Warm up on X_warm (earlier jets from the same detector), then correct X_raw block by block, each block
        with the estimate from the blocks before it. Returns the corrected copy of X_raw."""
        est = self.ref.copy()
        for X, correct in ((X_warm, False), (X_raw, True)):
            out = X.astype(np.float32).copy()
            for blk in np.array_split(np.arange(len(X)), max(1, len(X) // CAL_BLOCK)):
                if correct:
                    out[blk, self.cols[0]] *= self.ref[0] / est[0]
                    out[blk, self.cols[1]] -= np.round(est[1] - self.ref[1])
                v = X[blk][:, self.cols]
                est *= 1 + CAL_GAIN * ((v > est).mean(axis=0) - (v < est).mean(axis=0))
        self.est = est
        return out


def make_prep(flags, X_tr_raw, scaler, names):
    if flags.get("dedup") or flags.get("log"):
        return PhysicsScaler(dedup=flags.get("dedup", False), log=flags.get("log", False)).fit(X_tr_raw, names)
    return StandardPrep(scaler, names)

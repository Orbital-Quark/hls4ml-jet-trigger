import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

BACKGROUND = ("g", "q")      # ordinary QCD jets: what a trigger mostly sees
SIGNALS = ("t", "w", "z")    # what a jet trigger wants to keep
WORKING_RATE = 0.01          # background pass rate each trigger is allowed (1%)
ROC_RATES = np.logspace(-3, 0, 13)


def margin(logits, c):
    """Trigger score for class c: its score minus the best other class.
    On the chip this is a few subtractions and a max on the 5 raw outputs (softmax skipped)."""
    others = np.delete(logits, c, axis=1)
    return logits[:, c] - others.max(axis=1)


def cut_for_rate(bkg_scores, rate):
    """The threshold (pass = score >= cut) whose background pass rate is closest to `rate`.
    Low-bit circuits give many jets identical scores, so only some rates are reachable; the achieved
    rate is always measured and reported alongside the efficiency."""
    values = np.unique(bkg_scores)                      # every reachable cut, ascending
    s = np.sort(bkg_scores)
    rates = 1 - np.searchsorted(s, values, side="left") / len(s)  # pass rate at each cut
    return float(values[np.argmin(np.abs(rates - rate))])


class TriggerMenu:
    """Custom replacement for argmax decisions: one threshold per signal class, calibrated on
    validation jets so that exactly `rate` of the background passes. The data sets the thresholds."""

    def __init__(self, classes, rate=WORKING_RATE):
        self.classes, self.rate = list(classes), rate
        self.bkg = [self.classes.index(c) for c in BACKGROUND]

    def calibrate(self, logits, y):
        is_bkg = np.isin(y, self.bkg)
        self.thresholds = {s: cut_for_rate(margin(logits[is_bkg], self.classes.index(s)), self.rate)
                           for s in SIGNALS}
        return self

    def evaluate(self, logits, y):
        """Signal efficiency and background pass rate for each trigger, at the calibrated thresholds."""
        is_bkg = np.isin(y, self.bkg)
        out = {}
        for s, t in self.thresholds.items():
            c = self.classes.index(s)
            passed = margin(logits, c) >= t
            out[f"eff_{s}"] = float(passed[y == c].mean())
            out[f"bkg_{s}"] = float(passed[is_bkg].mean())
        return out


BLOCK = 256          # jets per rate check; a power of two, so the pass count is a small counter on the chip
GAIN = 0.01          # how far one check moves the threshold, as a fraction of the background score spread
PASSES = 10          # the stream is replayed this many times (~100k jets, about a millisecond of LHC jets)
SIGNAL_SHARE = 0.001  # the stream is mostly background, like real collisions; this share is signal jets


class SelfCorrectingMenu(TriggerMenu):
    """v4: each trigger keeps its own pass rate on budget while the detector drifts.

    The thresholds start where TriggerMenu calibrated them on the healthy detector. Every BLOCK jets, each
    trigger compares how many jets it passed with how many it should have, and nudges its threshold: up if
    it passed too many, down if too few. It needs no labels, only its own decisions, so it can run on the
    chip: one counter, one adder and one register per trigger, next to the comparator it already has.
    """

    def self_correct(self, logits_healthy, logits_drifted, y, rng, trace=False):
        """Run the controller on a drifted stream of the given jets, then measure on held-out jets.
        Returns, per signal: the budget (background rate before the drift), and background rate and efficiency
        with the fixed threshold, the self-corrected one, and a fresh labelled recalibration to the same budget
        (the best any threshold could do)."""
        out = {}
        for s, t0 in self.thresholds.items():
            c = self.classes.index(s)
            healthy, drifted = margin(logits_healthy, c), margin(logits_drifted, c)
            bkg, sig = np.flatnonzero(np.isin(y, self.bkg)), np.flatnonzero(y == c)
            rng.shuffle(bkg)
            rng.shuffle(sig)
            stream_b, test_b = bkg[: len(bkg) // 2], bkg[len(bkg) // 2:]
            n_sig = int(round(len(stream_b) * SIGNAL_SHARE / (1 - SIGNAL_SHARE)))
            stream, test_s = np.concatenate([stream_b, sig[:n_sig]]), sig[n_sig:]
            target = float((healthy[stream] >= t0).mean())  # the rate budget, set while the detector was healthy
            lo, hi = np.quantile(healthy[bkg], [0.1, 0.9])
            t, thr = t0, []
            for _ in range(PASSES):
                order = rng.permutation(stream)
                for blk in np.array_split(order, max(1, len(order) // BLOCK)):
                    t += GAIN * (hi - lo) * ((drifted[blk] >= t).mean() - target) / target
                    thr.append(t)
            # the budget each trigger is held to: its own background rate on these held-out jets before the drift
            out[f"bkg_budget_{s}"] = float((healthy[test_b] >= t0).mean())
            t_oracle = cut_for_rate(drifted[test_b], out[f"bkg_budget_{s}"])
            for tag, cut in (("fixed", t0), ("adapt", t), ("oracle", t_oracle)):
                out[f"bkg_{tag}_{s}"] = float((drifted[test_b] >= cut).mean())
                out[f"eff_{tag}_{s}"] = float((drifted[test_s] >= cut).mean())
            if trace:
                out[f"trace_{s}"] = [round(float(v), 5) for v in thr]
                out[f"trace_rate_{s}"] = [round(float((drifted[test_b] >= v).mean()) / out[f"bkg_budget_{s}"], 4)
                                          for v in thr[::4]]  # background rate as a multiple of the budget
        return out


def roc(logits, y, classes):
    """AUC and the ROC curve (signal efficiency at fixed background rates) for each signal vs q+g."""
    classes = list(classes)
    is_bkg = np.isin(y, [classes.index(c) for c in BACKGROUND])
    out = {}
    for s in SIGNALS:
        c = classes.index(s)
        sel = is_bkg | (y == c)
        score = margin(logits[sel], c)
        out[f"auc_{s}"] = float(roc_auc_score(y[sel] == c, score))
        fpr, tpr, _ = roc_curve(y[sel] == c, score)       # ties become straight segments (standard ROC)
        out[f"roc_{s}"] = [float(e) for e in np.interp(ROC_RATES, fpr, tpr)]
        out[f"eff1_{s}"] = float(np.interp(WORKING_RATE, fpr, tpr))  # efficiency at exactly 1% background
    return out

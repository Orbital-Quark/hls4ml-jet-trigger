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

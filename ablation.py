"""Which v2 change helps? Train one model (6-bit, full size, unpruned) with v1, each change on its own,
and all of v2. Keras-only (no synthesis), so the whole table takes minutes, not hours."""
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pandas as pd

from common import classes
from drift import apply_drift
from pipeline import V2, build_data, ensure_teacher
from train import train
from trigger import TriggerMenu

CFG = dict(bits=6, scale=1.0, sparsity=0.0)
VARIANTS = {"v1": {}, "+dedup": dict(dedup=True), "+log scaler": dict(log=True), "+augment": dict(augment=True),
            "+distill": dict(distill=True), "v2 (all)": V2}
TESTS = [("mass_scale", 0.05), ("smear", 0.1), ("pileup", 5)]


def logits(model, X):
    return np.log(model.predict(X, batch_size=8192, verbose=0) + 1e-9)  # log-probabilities: same margins as logits


rows = []
for label, flags in VARIANTS.items():
    if flags.get("distill"):
        print(ensure_teacher(flags), flush=True)
    d = build_data(flags)
    model = train(CFG, d.X_fit, d.y_fit, d.X_val, d.y_val, targets=d.targets)
    cl = classes()
    Xe = d.prep.transform(d.X_te_raw)
    le = logits(model, Xe)
    menu = TriggerMenu(cl).calibrate(logits(model, d.X_val), d.y_val)
    row = dict(variant=label, inputs=len(d.prep.names), acc=(le.argmax(1) == d.y_te).mean(), **menu.evaluate(le, d.y_te))
    for kind, level in TESTS:
        Xd = d.prep.transform(apply_drift(d.X_te_raw, kind, level, d.std, np.random.default_rng(1234), d.names))
        ld = logits(model, Xd)
        row[f"acc_{kind}"] = (ld.argmax(1) == d.y_te).mean()
        if kind == "mass_scale":
            row.update({f"{k}_mass5": v for k, v in menu.evaluate(ld, d.y_te).items() if k.startswith("bkg")})
    rows.append(row)
    print(row, flush=True)

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/ablation.csv", index=False)
pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
print(df.to_string(index=False))

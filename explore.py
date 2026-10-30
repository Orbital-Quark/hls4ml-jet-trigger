import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load, DATA
from models import build_model

X_tr, X_te, y_tr, y_te, scaler, names = load()
classes = np.load(DATA / "classes.npy", allow_pickle=True)
pd.set_option("display.width", 150, "display.float_format", "{:.4g}".format)

# 5.1 output: what the data looks like
print(f"Train {X_tr.shape}, test {X_te.shape}, classes {list(classes)}\n")

first = pd.DataFrame(X_tr[:5], columns=names).T
first.columns = [f"jet{i} ({classes[y_tr[i]]})" for i in range(5)]
print("First 5 training jets, raw values (one column per jet):")
print(first, "\n")

print("Jets per class (train):")
print(pd.Series(classes[y_tr]).value_counts().sort_index(), "\n")

df = pd.DataFrame(X_tr, columns=names)
df["class"] = classes[y_tr]
print("Median of each feature per class:")
print(df.groupby("class").median().T, "\n")

# What the scaler does to jet 0
X0 = scaler.transform(X_tr[:1])[0]
print("Scaling jet 0 (raw -> scaled):")
for n in ("mass_mmdt", "multiplicity"):
    i = names.index(n)
    print(f"  {n:13s} {X_tr[0, i]:8.2f} -> {X0[i]:6.2f}   (train mean {scaler.mean_[i]:.2f}, std {scaler.scale_[i]:.2f})")
print()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, n, rng in zip(axes, ("mass_mmdt", "multiplicity"), ((0, 250), (0, 120))):
    i = names.index(n)
    for k, c in enumerate(classes):
        ax.hist(X_tr[y_tr == k, i], bins=60, range=rng, histtype="step", density=True, label=c)
    ax.set_xlabel(n)
    ax.set_ylabel("fraction of jets")
    ax.legend(title="class")
fig.tight_layout()
fig.savefig("figures/explore_features.png", dpi=120)
print("Saved figures/explore_features.png\n")

# 5.3 output: the model, before any training
model = build_model(6, 1.0)
model.summary()
probs = model.predict(scaler.transform(X_tr[:3]), verbose=0)
print("\nUntrained 6-bit model, probabilities for the first 3 jets:")
print(pd.DataFrame(probs, columns=classes, index=[f"jet{i} ({classes[y_tr[i]]})" for i in range(3)]))
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from drift import DRIFTS
from trigger import ROC_RATES, SIGNALS, WORKING_RATE

Z7020 = dict(lut=53_200, ff=106_400, dsp=220, bram18=280)  # xc7z020: the chip the designs are synthesised for
VU9P = dict(lut=1_182_240, dsp=6_840)                        # XCVU9P: L1-trigger-class FPGA, shown for context

# Colours: bit width is ordered, so one blue ramp (light = fewer bits); mixed widths and versions get their own hues
BITS_ORDER = ["4", "6", "8", "16", "float", "8-4-4-8", "8-6-6-8"]
BITS_COLOR = {"4": "#86b6ef", "6": "#5598e7", "8": "#2a78d6", "16": "#1c5cab", "float": "#104281",
              "8-4-4-8": "#eb6834", "8-6-6-8": "#1baf7a"}
BITS_MARKER = {b: ("D" if "-" in b else "o") for b in BITS_ORDER}
VERSION_COLOR = {"v1": "#2a78d6", "v2": "#eb6834"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5})

Path("figures").mkdir(exist_ok=True)
df = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob("results/points/*.json"))])
for k, n in Z7020.items():
    df[f"{k}_pct_z7020"] = 100 * df[k] / n
df["fits_z7020"] = np.all([df[k] <= n for k, n in Z7020.items()], axis=0)
for k, n in VU9P.items():
    df[f"{k}_pct_vu9p"] = 100 * df[k] / n
drift_cols = [f"drift_{k}_{l}" for k, l in DRIFTS[1:]]
df["mean_drift_drop"] = 100 * (df["acc_hls"].to_numpy()[:, None] - df[drift_cols].to_numpy()).mean(axis=1)
for s in SIGNALS:  # background passing at the clean threshold under a 5% mass drift, relative to the clean rate
    df[f"rate_x_{s}_mass5"] = df[f"bkg_{s}_mass_scale_0.05"] / df[f"bkg_{s}"].where(df[f"bkg_{s}"] > 0)
df.drop(columns=[c for c in df.columns if c.startswith("roc_") or c == "thresholds"]).to_csv(
    "results/results.csv", index=False)
versions = sorted(df["version"].unique())


def pareto(d, x, y):
    d, keep, best = d.sort_values(x), [], -np.inf
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return d.loc[keep]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}.png", dpi=200)
    plt.close(fig)


# Fig 1-2: accuracy vs cost, one panel per version, coloured by bit width
for x, fname, label, limit in [("latency_ns", "fig1_accuracy_vs_latency", "Latency (ns)", None),
                               ("lut", "fig2a_accuracy_vs_lut", "LUTs", Z7020["lut"]),
                               ("dsp", "fig2b_accuracy_vs_dsp", "DSPs", Z7020["dsp"])]:
    fig, axes = plt.subplots(1, len(versions), figsize=(6 * len(versions), 4.5), sharex=True, sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):
        d = df[df["version"] == v]
        for b in [b for b in BITS_ORDER if b in set(d["bits_label"])]:
            g = d[d["bits_label"] == b]
            ax.scatter(g[x], g["acc_hls"], s=45, c=BITS_COLOR[b], marker=BITS_MARKER[b],
                       edgecolors=SURFACE, linewidths=1.5, label=f"{b}-bit" if b[0].isdigit() and "-" not in b else b)
        p = pareto(d, x, "acc_hls")
        ax.plot(p[x], p["acc_hls"], color=INK, ls="--", lw=1.2, label="Pareto frontier")
        if limit:
            ax.axvline(limit, color=INK2, lw=1, ls=":")
            ax.text(limit, ax.get_ylim()[0], " xc7z020 limit", color=INK2, fontsize=8, va="bottom")
            ax.set_xscale("symlog", linthresh=1)
        ax.set(title=v, xlabel=label)
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("Accuracy (circuit emulation)")
    save(fig, fname)

# Fig 1b: the two Pareto frontiers on one plot (v1 vs v2)
fig, ax = plt.subplots(figsize=(7, 4.5))
for v in versions:
    p = pareto(df[df["version"] == v], "dsp", "acc_hls")
    ax.plot(p["dsp"], p["acc_hls"], color=VERSION_COLOR[v], marker="o", ms=6, label=v)
ax.axvline(Z7020["dsp"], color=INK2, lw=1, ls=":")
ax.set(xscale="symlog", xlabel="DSPs", ylabel="Best accuracy at this DSP budget", title="Pareto frontier: v1 vs v2")
ax.legend(frameon=False)
save(fig, "fig1b_pareto_v1_vs_v2")

# Fig 3: accuracy drop under each drift, by bit width and by sparsity, one panel per version
for group in ("bits_label", "sparsity"):
    fig, axes = plt.subplots(len(versions), 1, figsize=(10, 2.2 + 1.6 * len(versions)), squeeze=False)
    drops = pd.DataFrame({c: 100 * (df["acc_hls"] - df[c]) for c in drift_cols})
    vmax = drops.groupby([df["version"], df[group]]).mean().to_numpy().max()
    for ax, v in zip(axes[:, 0], versions):
        sel = df["version"] == v
        heat = drops[sel].groupby(df.loc[sel, group]).mean()
        if group == "bits_label":
            heat = heat.reindex([b for b in BITS_ORDER if b in heat.index])
        im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=BLUES, vmin=0, vmax=vmax)
        for (i, j), val in np.ndenumerate(heat.to_numpy()):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if val > 0.6 * vmax else INK)
        ax.set_xticks(range(len(drift_cols)), [c.replace("drift_", "") for c in drift_cols], rotation=30, ha="right")
        ax.set_yticks(range(len(heat.index)), [str(i) for i in heat.index])
        ax.set_title(f"{v}: accuracy drop under drift (percentage points), by {group}", fontsize=10)
        ax.grid(False)
    fig.colorbar(im, ax=axes[:, 0].tolist(), label="Accuracy drop (points)")
    fig.savefig(f"figures/fig3_robustness_by_{group}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Fig 4: trigger performance vs cost: top efficiency at 1% background, v1 vs v2
fig, ax = plt.subplots(figsize=(7, 4.5))
for v in versions:
    d = df[df["version"] == v]
    ax.scatter(d["dsp"], d["eff1_t"], s=40, c=VERSION_COLOR[v], edgecolors=SURFACE, linewidths=1.5, label=v)
ax.axvline(Z7020["dsp"], color=INK2, lw=1, ls=":")
ax.set(xscale="symlog", xlabel="DSPs", ylabel=f"Top efficiency at {WORKING_RATE:.0%} q/g background",
       title="Trigger performance vs hardware cost")
ax.legend(frameon=False)
save(fig, "fig4_top_efficiency_vs_dsp")

# Fig 5: rate blow-up under a 5% mass drift (thresholds fixed on the healthy detector), by bit width
fig, axes = plt.subplots(1, len(SIGNALS), figsize=(13, 4), sharey=True)
for ax, s in zip(axes, SIGNALS):
    m = df.groupby(["bits_label", "version"])[f"rate_x_{s}_mass5"].mean().unstack()
    m = m.reindex([b for b in BITS_ORDER if b in m.index])
    w = 0.8 / len(versions)
    for k, v in enumerate(versions):
        if v in m:
            ax.bar(np.arange(len(m)) + (k - (len(versions) - 1) / 2) * w, m[v], width=w - 0.04,
                   color=VERSION_COLOR[v], label=v)
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(m)), m.index, rotation=30)
    ax.set_title(f"{s} trigger")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Background rate, drifted ÷ healthy\n(5% mass drift, same threshold)")
axes[0].legend(frameon=False)
save(fig, "fig5_rate_under_mass_drift")

# Fig 6: ROC curves (top vs q/g) for representative full-size, unpruned designs
fig, ax = plt.subplots(figsize=(7, 4.5))
rep = df[(df["scale"] == 1.0) & (df["sparsity"] == 0.0) & (df["rf"] == df["rf"].max())
         & df["bits_label"].isin(["float", "8", "4"])]
for _, r in rep.sort_values(["version", "bits_label"]).iterrows():
    ax.plot(ROC_RATES, r["roc_t"], color=VERSION_COLOR[r["version"]],
            ls={"float": "-", "8": "--", "4": ":"}[r["bits_label"]], label=f"{r['version']} {r['bits_label']}")
ax.axvline(WORKING_RATE, color=INK2, lw=1, ls=":")
ax.set(xscale="log", xlabel="q/g background pass rate", ylabel="Top efficiency", title="ROC: top vs q/g")
ax.legend(frameon=False, fontsize=8)
save(fig, "fig6_roc_top")

# v1 vs v2 for the same design (bits, size, pruning, reuse factor)
if {"v1", "v2"} <= set(versions):
    key = ["bits_label", "scale", "sparsity", "rf"]
    cols = ["acc_hls", "mean_drift_drop", "eff1_t", "rate_x_t_mass5", "dsp", "lut"]
    both = df[df["version"] == "v1"][key + cols].merge(df[df["version"] == "v2"][key + cols], on=key,
                                                      suffixes=("_v1", "_v2"))
    for c in cols:
        both[f"d_{c}"] = both[f"{c}_v2"] - both[f"{c}_v1"]
    both.to_csv("results/v1_vs_v2.csv", index=False)
    print("v2 minus v1, averaged over matched designs:")
    print(both[[f"d_{c}" for c in cols]].mean().round(4).to_string())

pd.set_option("display.width", 200)
print(df.sort_values(["version", "latency_ns"])[["name", "acc_keras", "acc_hls", "eff1_t", "latency_ns", "ii", "dsp",
                                                 "lut", "fits_z7020", "mean_drift_drop"]].to_string(index=False))

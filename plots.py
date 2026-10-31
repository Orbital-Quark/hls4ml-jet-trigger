"""Graphs from the sweep, in the hls4ml paper's style: change one setting at a time, one line per version.
Saves fig*.png / paper*.png and a gallery page index.html with every figure and what it shows.
  python plots.py                 # one set per version: figures/v1/, figures/v2/, ...
  python plots.py v4              # just v4 -> figures/v4/
  python plots.py v1 v2 v4        # these versions compared on the same graphs -> figures/v1_v2_v4/
  add --open to open the gallery in the Windows browser"""
import glob
import html
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analyze import DRIFT_COLS, MIN_ACC, Z7020, load_results
from drift import DRIFTS
from trigger import ROC_RATES, SIGNALS, WORKING_RATE

# Colour = version (fixed order); line style = the second setting in a plot
BITS_SCAN = ["4", "6", "8", "16", "float"]
BITS_TICKS = ["4-bit", "6-bit", "8-bit", "16-bit", "float\n(16,6)"]
VERSION_COLOR = {"v1": "#2a78d6", "v2": "#eb6834", "v3": "#1baf7a", "v4": "#eda100"}
SIZE_STYLE = {1.0: "-", 0.5: "--", 0.25: ":"}
RF_STYLE = {1: "-", 4: "--"}
DRIFT_TITLE = {"mass_scale": "Mass scale shift (%)", "smear": "Resolution smearing (% of spread)",
               "pileup": "Extra pileup particles"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
                     "lines.markersize": 6, "font.size": 11, "axes.titlesize": 12, "legend.fontsize": 9,
                     "legend.frameon": False})

CAPTIONS = {
    "fig1_precision_scan": "How accuracy and top-trigger efficiency change with the number of bits (full size, unpruned).",
    "fig2_precision_hardware": "Multipliers (DSPs) and logic (LUTs) vs number of bits, for reuse 1 (solid, fastest) and "
                               "reuse 4 (dashed). Dotted line = what the xc7z020 has.",
    "fig3_pruning": "Left: accuracy vs pruning for three network sizes (6-bit). Right: pruning only removes multipliers "
                    "at reuse 1; at reuse 4 every weight keeps its slot.",
    "fig4_drift_scan": "Accuracy as each kind of detector drift grows, with the model and thresholds unchanged.",
    "fig5_pareto": "Every design (faint) and the best accuracy reachable for each hardware budget (line).",
    "fig6_rate_under_mass_drift": "How much background each trigger lets through after a 5% mass shift, relative to the "
                                  "healthy detector (1 = unchanged).",
    "fig7_roc_top": "Top efficiency at each background pass rate (ROC). Dotted line = the 1% working point.",
    "fig8_drift_heatmap": "Accuracy lost under every drift level, averaged over all designs of each precision.",
    "paper1_auc_ratio": "Paper style (hls4ml paper Fig. 9): each tagger's AUC relative to the float design, vs "
                        "precision. 1 = no loss of separation power.",
    "paper2_roc": "Paper style: ROC curves, signal efficiency vs background pass rate (log). Solid = float, "
                  "dashed = 6-bit.",
    "paper3_resources": "Paper style: DSPs, flip-flops and LUTs vs precision for each reuse factor (full size, "
                        "unpruned). Dotted line = what the xc7z020 has.",
    "paper4_latency": "Paper style: latency and initiation interval (how often a new jet can enter) vs precision "
                      "for each reuse factor.",
    "fig9_self_correcting": "v4 self-correcting trigger. Top: background rate as a multiple of the budget as each "
                            "drift grows, fixed threshold (dashed) vs self-correcting (solid). Bottom: signal kept, "
                            "fixed vs self-correcting vs a fresh labelled recalibration (best possible).",
    "fig10_self_correcting_trace": "v4 self-correcting trigger after a 10% mass-scale shift: background rate (held-out "
                                   "jets) as the controller sees more jets, vs the fixed threshold.",
}

picked = [a for a in sys.argv[1:] if not a.startswith("--")]
df = load_results()
if not picked:  # default: one set of graphs per version, each in its own folder
    for v in sorted(df["version"].unique()):
        subprocess.run([sys.executable, __file__, v] + [a for a in sys.argv[1:] if a.startswith("--")])
    sys.exit()
df = df[df["version"].isin(picked)].reset_index(drop=True)
OUT = Path("figures") / "_".join(picked)
ok = df[df["acc_hls"] >= MIN_ACC]
versions = sorted(df["version"].unique())
rf_lo = df["rf"].min()
x_bits = np.arange(len(BITS_SCAN))
OUT.mkdir(parents=True, exist_ok=True)
for old in glob.glob(str(OUT / "fig*.png")) + glob.glob(str(OUT / "paper*.png")):  # names change between script versions
    Path(old).unlink()


def pick(v, bits=None, scale=1.0, sparsity=0.0, rf=None):
    sel = (df["version"] == v) & (df["scale"] == scale) & (df["sparsity"] == sparsity) & (df["rf"] == (rf or rf_lo))
    if bits is not None:
        sel &= df["bits_label"] == bits
    return df[sel]


def bits_scan(v, col, **kw):
    return pick(v, **kw).set_index("bits_label").reindex(BITS_SCAN)[col].to_numpy(dtype=float)


def pareto(d, x, y):
    d, keep, best = d.sort_values(x), [], -np.inf
    for i, r in d.iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return d.loc[keep]


def chip_line(ax, value, label):
    ax.axhline(value, color=INK2, lw=1, ls=":")
    ax.annotate(label, (1, value), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8, color=INK2)


def save(fig, name, note="full size, unpruned, circuit emulation on 50k test jets"):
    if note:
        fig.text(0.99, 0.01, note, ha="right", va="bottom", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# Fig 1: what precision costs in accuracy and trigger efficiency
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for v in versions:
    axes[0].plot(x_bits, 100 * bits_scan(v, "acc_hls"), marker="o", color=VERSION_COLOR[v], label=v)
    axes[1].plot(x_bits, 100 * bits_scan(v, "eff1_t"), marker="o", color=VERSION_COLOR[v], label=v)
axes[0].set(ylabel="Accuracy (%)", title="(a) Accuracy vs precision")
axes[1].set(ylabel=f"Top efficiency at {WORKING_RATE:.0%} background (%)", title="(b) Top trigger vs precision")
for ax in axes:
    ax.set_xticks(x_bits, BITS_TICKS)
axes[0].legend()
save(fig, "fig1_precision_scan")

# Fig 2: what precision costs in hardware, for both design styles (reuse factor)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for v in versions:
    for rf in sorted(df["rf"].unique()):
        for ax, col in zip(axes, ("dsp", "lut")):
            ax.plot(x_bits, bits_scan(v, col, rf=rf), marker="o", color=VERSION_COLOR[v], ls=RF_STYLE.get(rf, "-"),
                    label=f"{v}, reuse {rf}")
chip_line(axes[0], Z7020["dsp"], "xc7z020: 220 DSPs")
chip_line(axes[1], Z7020["lut"], "xc7z020: 53,200 LUTs")
axes[0].set(ylabel="DSPs", yscale="symlog", title="(a) Multipliers (DSPs) vs precision")
axes[0].set_ylim(bottom=0)
axes[1].set(ylabel="LUTs", yscale="log", title="(b) Logic (LUTs) vs precision")
for ax in axes:
    ax.set_xticks(x_bits, BITS_TICKS)
axes[0].legend(ncol=2)
save(fig, "fig2_precision_hardware")

# Fig 3: pruning: accuracy by network size (6-bit), and whether it saves multipliers in each design style (8-bit)
prune = sorted(df["sparsity"].unique())
x_p = [100 * p for p in prune]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for v in versions:
    for s, ls in SIZE_STYLE.items():
        acc = [100 * pick(v, "6", s, p)["acc_hls"].mean() for p in prune]
        axes[0].plot(x_p, acc, marker="o", color=VERSION_COLOR[v], ls=ls, label=f"{v}, {int(s * 100)}% size")
    for rf in sorted(df["rf"].unique()):
        axes[1].plot(x_p, [pick(v, "8", 1.0, p, rf)["dsp"].mean() for p in prune], marker="o", color=VERSION_COLOR[v],
                     ls=RF_STYLE.get(rf, "-"), label=f"{v}, reuse {rf}")
axes[0].set(xlabel="Weights pruned (%)", ylabel="Accuracy (%)", title="(a) Accuracy vs pruning (6-bit)")
axes[0].set_ylim(bottom=100 * MIN_ACC - 5)
axes[1].set(xlabel="Weights pruned (%)", ylabel="DSPs", title="(b) Multipliers vs pruning (8-bit, full size)")
axes[1].set_ylim(bottom=0)
for ax in axes:
    ax.set_xticks(x_p)
    ax.legend(ncol=2 if len(versions) > 1 else 1)
save(fig, "fig3_pruning", note="circuit emulation on 50k test jets; points below the axis collapsed (see results.csv)")

# Fig 4: robustness: accuracy as each kind of detector drift grows
kinds = list(dict.fromkeys(k for k, _ in DRIFTS[1:]))
fig, axes = plt.subplots(1, len(kinds), figsize=(13, 4), sharey=True)
for ax, kind in zip(axes, kinds):
    levels = [0] + [l for k, l in DRIFTS[1:] if k == kind]
    x = [100 * l if kind != "pileup" else l for l in levels]
    for v in versions:
        for (bits, scale), ls, tag in [(("float", 1.0), "-", "float, full size"), (("6", 0.5), "--", "6-bit, half size")]:
            r = pick(v, bits, scale)
            if len(r):
                r = r.iloc[0]
                acc = [r["acc_hls"]] + [r[f"drift_{kind}_{l}"] for l in levels[1:]]
                ax.plot(x, 100 * np.array(acc), marker="o", color=VERSION_COLOR[v], ls=ls, label=f"{v}, {tag}")
    ax.set(xlabel=DRIFT_TITLE[kind], xticks=x)
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
fig.suptitle("Accuracy as the detector drifts (model and thresholds unchanged)", fontsize=12)
save(fig, "fig4_drift_scan", note="circuit emulation on 50k test jets")

# Fig 5: accuracy vs cost over every design: Pareto frontier per version, all designs faint behind it
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, col, label in zip(axes, ("dsp", "lut"), ("DSPs", "LUTs")):
    for v in versions:
        d = ok[ok["version"] == v]
        ax.scatter(d[col], 100 * d["acc_hls"], s=14, color=VERSION_COLOR[v], alpha=0.25, edgecolors="none")
        p = pareto(d, col, "acc_hls")
        ax.step(p[col], 100 * p["acc_hls"], where="post", color=VERSION_COLOR[v], marker="o", ms=4, label=v)
    ax.axvline(Z7020[col], color=INK2, lw=1, ls=":")
    ax.annotate("xc7z020 limit", (Z7020[col], 0.02), xycoords=("data", "axes fraction"), fontsize=8, color=INK2,
                ha="right", rotation=90)
    ax.set(xlabel=label, xscale="symlog" if col == "dsp" else "log")
    if col == "dsp":
        ax.set_xlim(left=0)
axes[0].set(ylabel="Accuracy (%)", title="(a) Best accuracy for a DSP budget")
axes[1].set(title="(b) Best accuracy for a LUT budget")
axes[0].legend(loc="lower right")
save(fig, "fig5_pareto", note=f"every design with accuracy ≥ {MIN_ACC:.0%} (faint), Pareto frontier (line)")

# Fig 6: trigger rate under a 5% mass drift (thresholds fixed on the healthy detector), by precision
fig, axes = plt.subplots(1, len(SIGNALS), figsize=(13, 4), sharey=True)
w = 0.8 / len(versions)
for ax, s in zip(axes, SIGNALS):
    m = ok.groupby(["bits_label", "version"])[f"rate_x_{s}_mass5"].mean().unstack().reindex(BITS_SCAN)
    for k, v in enumerate(versions):
        if v in m:  # a version can be missing while a sweep is still running
            ax.bar(x_bits + (k - (len(versions) - 1) / 2) * w, m[v], width=w - 0.04, color=VERSION_COLOR[v], label=v)
    ax.axhline(1, color=INK2, lw=1, ls=":")
    ax.set_xticks(x_bits, BITS_TICKS)
    ax.set_title(f"{s.upper()} trigger")
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Background rate, drifted ÷ healthy")
axes[0].legend()
fig.suptitle("Background rate after a 5% mass-scale shift (1 = unchanged)", fontsize=12)
save(fig, "fig6_rate_under_mass_drift", note="average over every design of each precision; circuit emulation")

# Fig 7: ROC curves (top vs q/g), float and 6-bit full-size designs
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for v in versions:
    for bits, ls in (("float", "-"), ("6", "--")):
        r = pick(v, bits)
        if len(r):
            ax.plot(ROC_RATES, 100 * np.array(r.iloc[0]["roc_t"]), color=VERSION_COLOR[v], ls=ls, label=f"{v}, {bits}")
ax.axvline(WORKING_RATE, color=INK2, lw=1, ls=":")
ax.set(xscale="log", xlabel="q/g background pass rate", ylabel="Top efficiency (%)", title="ROC: top vs q/g")
ax.legend()
save(fig, "fig7_roc_top")

# Fig 8: accuracy lost under each drift, by precision, one row per version
heat_all = pd.DataFrame({c: 100 * (ok["acc_hls"] - ok[c]) for c in DRIFT_COLS})
vmax = np.nanmax(heat_all.groupby([ok["version"], ok["bits_label"]]).mean().to_numpy())
fig, axes = plt.subplots(len(versions), 1, figsize=(10, 1.2 + 1.9 * len(versions)), squeeze=False)
for ax, v in zip(axes[:, 0], versions):
    sel = ok["version"] == v
    heat = heat_all[sel].groupby(ok.loc[sel, "bits_label"]).mean().reindex(BITS_SCAN)
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=BLUES, vmin=0, vmax=vmax)
    for (i, j), val in np.ndenumerate(heat.to_numpy()):
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=SURFACE if val > 0.6 * vmax else INK)
    ax.set_xticks(range(len(DRIFT_COLS)), [c.replace("drift_", "").replace("_scale", "") for c in DRIFT_COLS],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(BITS_SCAN)), BITS_SCAN)
    ax.set_title(f"{v}: accuracy lost (points)", fontsize=11)
    ax.grid(False)
fig.colorbar(im, ax=axes[:, 0].tolist(), label="Accuracy lost (points)")
fig.savefig(OUT / "fig8_drift_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Paper-style figures: the layouts of the hls4ml paper (arXiv:1804.06913) with its look: boxed axes, inward ticks,
# framed legends and a label in the corner. Colour = tagger or reuse factor here, one panel per version.
PAPER_RC = {"axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": INK, "xtick.direction": "in",
            "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.color": INK, "ytick.color": INK,
            "grid.linestyle": "--", "grid.color": "#d0d0d0", "legend.frameon": True, "legend.edgecolor": "#9a9a9a",
            "legend.fancybox": False}
TAGGER_COLOR = {"t": "#2a78d6", "w": "#eb6834", "z": "#1baf7a"}
RF_COLOR = {1: "#1c5cab", 4: "#86b6ef"}
VERSION_STYLE = {"v1": "-", "v2": "--", "v3": ":", "v4": "-."}
rfs = sorted(df["rf"].unique())


def corner(ax, text):
    ax.text(0.0, 1.015, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")


with plt.rc_context(PAPER_RC):
    # Paper 1: AUC relative to the float design vs precision, one line per tagger (paper Fig. 9)
    fig, axes = plt.subplots(1, len(versions), figsize=(5.2 * len(versions), 4.2), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):
        for s in SIGNALS:
            auc = bits_scan(v, f"auc_{s}")
            ax.plot(x_bits, auc / auc[-1], marker="s", ms=5, color=TAGGER_COLOR[s], label=f"{s} tagger")
        ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set_xticks(x_bits, BITS_TICKS)
        ax.set(xlabel="Fixed-point precision")
        corner(ax, f"hls4ml  ·  {v}")
    axes[0][0].set_ylabel("AUC / AUC of float design")
    axes[0][0].legend(loc="lower right")
    save(fig, "paper1_auc_ratio")

    # Paper 2: ROC curves, signal efficiency vs background pass rate on a log axis, per tagger
    fig, axes = plt.subplots(1, len(versions), figsize=(5.2 * len(versions), 4.4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):
        for s in SIGNALS:
            for bits, ls in (("float", "-"), ("6", "--")):
                r = pick(v, bits)
                if len(r):
                    ax.plot(r.iloc[0][f"roc_{s}"], ROC_RATES, color=TAGGER_COLOR[s], ls=ls,
                            label=f"{s} tagger, {'float' if bits == 'float' else '6-bit'}")
        ax.axhline(WORKING_RATE, color=INK2, lw=1, ls=":")
        ax.set(xlabel="Signal efficiency", yscale="log", xlim=(0, 1), ylim=(ROC_RATES[0], 1))
        corner(ax, f"hls4ml  ·  {v}")
    axes[0][0].set_ylabel("Background (q/g) pass rate")
    axes[0][0].legend(loc="lower right", fontsize=8)
    save(fig, "paper2_roc")

    # Paper 3: resources vs precision, one line per reuse factor (paper Figs. 11-12), one row per version
    res = [("dsp", "DSPs", Z7020["dsp"]), ("ff", "Flip-flops", Z7020["ff"]), ("lut", "LUTs", Z7020["lut"])]
    fig, axes = plt.subplots(len(versions), 3, figsize=(14, 3.6 * len(versions)), squeeze=False)
    for row, v in zip(axes, versions):
        for ax, (col, label, limit) in zip(row, res):
            for rf in rfs:
                ax.plot(x_bits, bits_scan(v, col, rf=rf), marker="o", color=RF_COLOR.get(rf, INK), label=f"reuse {rf}")
            ax.axhline(limit, color=INK2, lw=1, ls=":")
            ax.set_xticks(x_bits, BITS_TICKS)
            ax.set(ylabel=label, yscale="symlog" if col == "dsp" else "log")
            if col == "dsp":
                ax.set_ylim(bottom=0)
            corner(ax, f"hls4ml  ·  {v}")
        row[0].legend(loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Fixed-point precision")
    save(fig, "paper3_resources")

    # Paper 4: latency and initiation interval vs precision, colour = reuse factor, line style = version
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v in versions:
        for rf in rfs:
            axes[0].plot(x_bits, bits_scan(v, "latency_ns", rf=rf), marker="o", color=RF_COLOR.get(rf, INK),
                         ls=VERSION_STYLE.get(v, "-"), label=f"{v}, reuse {rf}")
            axes[1].plot(x_bits, 5 * bits_scan(v, "ii", rf=rf), marker="o", color=RF_COLOR.get(rf, INK),
                         ls=VERSION_STYLE.get(v, "-"), label=f"{v}, reuse {rf}")
    axes[1].axhline(25, color=INK2, lw=1, ls=":")
    axes[1].annotate("LHC bunch spacing: 25 ns", (1, 25), xycoords=("axes fraction", "data"), ha="right",
                     va="bottom", fontsize=8, color=INK2)
    axes[0].set(ylabel="Latency (ns)")
    axes[1].set(ylabel="Initiation interval (ns)")
    for ax in axes:
        ax.set_xticks(x_bits, BITS_TICKS)
        ax.set_xlabel("Fixed-point precision")
        ax.set_ylim(bottom=0)
        corner(ax, "hls4ml  ·  xc7z020, 200 MHz")
    axes[0].legend(ncol=2, fontsize=8)
    save(fig, "paper4_latency")

# Fig 9-10: v4's self-correcting trigger, averaged over every v4 design (colour = trigger)
sc = ok[ok.filter(like="bkg_adapt_").notna().any(axis=1)] if any(c.startswith("bkg_adapt_") for c in df) else ok[:0]
if len(sc):
    from trigger import BLOCK
    fig, axes = plt.subplots(2, len(kinds), figsize=(13, 7.5), sharey="row")
    for j, kind in enumerate(kinds):
        levels = [l for k, l in DRIFTS[1:] if k == kind]
        x = [100 * l if kind != "pileup" else l for l in levels]
        for s in SIGNALS:
            for tag, ls in (("fixed", "--"), ("adapt", "-")):
                rate = [(sc[f"bkg_{tag}_{s}_{kind}_{l}"] / sc[f"bkg_budget_{s}_{kind}_{l}"]).mean() for l in levels]
                axes[0, j].plot(x, rate, marker="o", color=TAGGER_COLOR[s], ls=ls,
                                label=f"{s.upper()}, {'fixed' if tag == 'fixed' else 'self-correcting'}")
        for tag, ls, lab in (("fixed", "--", "fixed"), ("adapt", "-", "self-correcting"), ("oracle", ":", "best")):
            eff = [100 * np.mean([sc[f"eff_{tag}_{s}_{kind}_{l}"].mean() for s in SIGNALS]) for l in levels]
            axes[1, j].plot(x, eff, marker="o", color=INK if tag != "adapt" else VERSION_COLOR["v4"], ls=ls, label=lab)
        axes[0, j].axhline(1, color=INK2, lw=1, ls=":")
        axes[0, j].set_title(DRIFT_TITLE[kind])
        axes[1, j].set(xlabel=DRIFT_TITLE[kind], xticks=x)
        axes[0, j].set_xticks(x)
    axes[0, 0].set_ylabel("Background rate ÷ budget")
    axes[1, 0].set_ylabel("Signal kept (%, average of t/w/z)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].legend()
    fig.suptitle("Self-correcting trigger (v4): rate held on budget as the detector drifts", fontsize=12)
    save(fig, "fig9_self_correcting", note=f"average over {len(sc)} v4 designs; held-out jets; circuit emulation")

    col = [c for c in sc.columns if c.startswith("trace_rate_") and c.endswith("mass_scale_0.1")]
    r = (pick("v4", "float") if len(pick("v4", "float")) else sc).iloc[0]
    if col and isinstance(r[col[0]], list):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for s in SIGNALS:
            tr = np.array(r[f"trace_rate_{s}_mass_scale_0.1"])  # already a multiple of the budget
            jets = (np.arange(len(tr)) + 1) * 4 * BLOCK
            ax.plot(jets, tr, color=TAGGER_COLOR[s], label=f"{s.upper()}, self-correcting")
            ax.axhline(r[f"bkg_fixed_{s}_mass_scale_0.1"] / r[f"bkg_budget_{s}_mass_scale_0.1"], color=TAGGER_COLOR[s],
                       ls="--", lw=1.2, label=f"{s.upper()}, fixed")
        ax.axhline(1, color=INK2, lw=1, ls=":")
        ax.set(xlabel="Jets seen by the controller", ylabel="Background rate ÷ budget",
               title=f"After a 10% mass-scale shift ({r['name']})")
        ax.legend(ncol=2, fontsize=8)
        save(fig, "fig10_self_correcting_trace", note="background rate measured on held-out jets; circuit emulation")

# Gallery page: every figure with its caption, so the graphs can be viewed next to the terminal output
cards = "\n".join(f'<figure><img src="{n}.png" alt="{html.escape(c)}"><figcaption><b>{n}</b> — {html.escape(c)}'
                  f"</figcaption></figure>" for n, c in CAPTIONS.items() if (OUT / f"{n}.png").exists())
live = "--live" in sys.argv  # during a sweep: the page reloads itself every minute
(OUT / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Sweep figures</title>
{'<meta http-equiv="refresh" content="60"><p><b>Sweep running:</b> this page updates itself every minute.</p>' if live else ''}
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;background:#fcfcfb;color:#0b0b0b}}
figure{{margin:0 0 32px}}img{{width:100%;border:1px solid #e4e3df}}figcaption{{color:#52514e;margin-top:6px}}</style>
<h1>Sweep figures: {", ".join(versions)}</h1>
<p>{len(df)} designs, {int(df["fits_z7020"].sum())} fit the xc7z020. Left out of the summary plots (collapsed,
accuracy &lt; {MIN_ACC:.0%}): {len(df) - len(ok)}.</p>
{cards}""")
print(f"saved {len(CAPTIONS)} figures and {OUT / 'index.html'}")

if "--open" in sys.argv:  # WSL: hand the page to Windows, which opens it in the default browser
    win = subprocess.run(["wslpath", "-w", str((OUT / "index.html").resolve())], capture_output=True, text=True)
    subprocess.run(["explorer.exe", win.stdout.strip()])

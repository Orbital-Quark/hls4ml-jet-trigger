"""Interactive results page for the sweep: figures/report.html, drawn in the browser from figures/data.js.
  python plots.py                # every version in results/: one tab each, plus a Compare tab
  python plots.py v1 v4          # only these versions
  --open    open the page in the Windows browser
  --live    the page re-reads the data every 30 s (used while a sweep runs)
  --tab=v4  the tab shown first
  --png     also save the paper-style images for the written report (figures/png/)"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analyze import CLOCK_NS, DRIFT_COLS, MIN_ACC, Z7020, load_results
from drift import DRIFTS
from trigger import ROC_RATES, SIGNALS, WORKING_RATE

OUT = Path("figures")
PAGE = Path(__file__).with_name("report.html")  # the page itself; it reads its numbers from data.js next to it
VERSION_COLOR = {"v1": "#2a78d6", "v2": "#eb6834", "v3": "#1baf7a", "v4": "#eda100", "v5": "#8b5cd6", "v5.2": "#d6457d"}
# hls4ml paper (Duarte et al., JINST 13 P07027, arXiv:1804.06913), same dataset and network: float AUCs (Fig. 5) and
# its benchmark design, 70% pruned, <16,6>, reuse 1, 200 MHz on a Kintex UltraScale xcku115 (Table 2)
PAPER = dict(auc=dict(t=0.958, w=0.946, z=0.939, g=0.938, q=0.904), dsp=954, chip_dsp=5520, logic=88_797,
             latency_ns=75, source="hls4ml paper, arXiv:1804.06913")
DRIFT_LABEL = {"mass_scale": "mass +{:.0%}", "smear": "smear {:.0%}", "pileup": "pileup +{}"}
KEEP = ["name", "version", "bits_label", "scale", "sparsity", "rf", "acc_hls", "auc_t", "auc_w", "auc_z", "eff1_t",
        "eff1_w", "eff1_z", "dsp", "lut", "ff", "latency_ns", "ii", "fits_z7020", "mean_drift_drop"]


def self_correcting(d):
    """v4: background rate as a multiple of the budget, and signal kept, per drift, averaged over the designs."""
    col = "bkg_adapt_t_mass_scale_0.1"
    d = d[d[col].notna()] if col in d else d[:0]
    if not len(d):
        return []
    rows = []
    for kind, level in DRIFTS[1:]:
        k = f"{kind}_{level}"
        ratio = lambda tag, s: float((d[f"bkg_{tag}_{s}_{k}"] / d[f"bkg_budget_{s}_{k}"]).replace(np.inf, np.nan).mean())
        rows.append(dict(key=k, n=len(d),
                         rate={tag: {s: ratio(tag, s) for s in SIGNALS} for tag in ("fixed", "adapt", "oracle")},
                         eff={tag: {s: float(d[f"eff_{tag}_{s}_{k}"].mean()) for s in SIGNALS}
                              for tag in ("fixed", "adapt", "oracle")}))
    return rows


def payload(df, live, focus):
    d = df[KEEP].copy()
    for (kind, level), col in zip(DRIFTS[1:], DRIFT_COLS):
        d[f"loss_{kind}_{level}"] = 100 * (df["acc_hls"] - df[col])
    ok = df[df["acc_hls"] >= MIN_ACC]
    versions = sorted(df["version"].unique())
    sc = {v: self_correcting(ok[ok["version"] == v]) for v in versions}
    return dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), live=live, focus=focus, versions=versions,
                colors=VERSION_COLOR, chip=Z7020, clock_ns=CLOCK_NS, min_acc=MIN_ACC, paper=PAPER,
                drifts=[dict(key=f"{k}_{l}", label=DRIFT_LABEL[k].format(l)) for k, l in DRIFTS[1:]],
                designs=json.loads(d.to_json(orient="records")), self_correcting={v: r for v, r in sc.items() if r})


def write_pngs(df, out):
    """Static figures in the hls4ml paper's layouts and look, for the written report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    bits_scan_order, ticks = ["4", "6", "8", "16", "float"], ["4-bit", "6-bit", "8-bit", "16-bit", "float\n(16,6)"]
    x_bits, rfs, versions = np.arange(5), sorted(df["rf"].unique()), sorted(df["version"].unique())
    ok = df[df["acc_hls"] >= MIN_ACC]
    ink, ink2 = "#0b0b0b", "#52514e"
    tagger = {"t": "#2a78d6", "w": "#eb6834", "z": "#1baf7a"}
    rf_color = {1: "#1c5cab", 4: "#86b6ef"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": ink,
                         "text.color": ink, "axes.labelcolor": ink, "xtick.color": ink, "ytick.color": ink,
                         "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
                         "axes.grid": True, "grid.linestyle": "--", "grid.color": "#d0d0d0", "lines.linewidth": 1.8,
                         "font.size": 11, "legend.fontsize": 9, "legend.edgecolor": "#9a9a9a", "legend.fancybox": False})

    def pick(v, bits, rf=None):
        sel = (df["version"] == v) & (df["scale"] == 1.0) & (df["sparsity"] == 0.0) & (df["rf"] == (rf or rfs[0]))
        return df[sel & (df["bits_label"] == bits)]

    def scan(v, col, rf=None):
        sel = (df["version"] == v) & (df["scale"] == 1.0) & (df["sparsity"] == 0.0) & (df["rf"] == (rf or rfs[0]))
        return df[sel].set_index("bits_label").reindex(bits_scan_order)[col].to_numpy(dtype=float)

    def corner(ax, text):
        ax.text(0.0, 1.015, text, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(versions), figsize=(5.2 * len(versions), 4.2), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):  # paper Fig. 9: AUC relative to the float design vs precision
        for s in SIGNALS:
            auc = scan(v, f"auc_{s}")
            ax.plot(x_bits, auc / auc[-1], marker="s", ms=5, color=tagger[s], label=f"{s} tagger")
        ax.axhline(1, color=ink2, lw=1, ls=":")
        ax.set_xticks(x_bits, ticks)
        ax.set(xlabel="Fixed-point precision")
        corner(ax, f"hls4ml  ·  {v}")
    axes[0][0].set_ylabel("AUC / AUC of float design")
    axes[0][0].legend(loc="lower right")
    save(fig, "paper1_auc_ratio")

    fig, axes = plt.subplots(1, len(versions), figsize=(5.2 * len(versions), 4.4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], versions):  # ROC curves, background on a log axis
        for s in SIGNALS:
            for bits, ls in (("float", "-"), ("6", "--")):
                r = pick(v, bits)
                if len(r):
                    ax.plot(r.iloc[0][f"roc_{s}"], ROC_RATES, color=tagger[s], ls=ls,
                            label=f"{s} tagger, {'float' if bits == 'float' else '6-bit'}")
        ax.axhline(WORKING_RATE, color=ink2, lw=1, ls=":")
        ax.set(xlabel="Signal efficiency", yscale="log", xlim=(0, 1), ylim=(ROC_RATES[0], 1))
        corner(ax, f"hls4ml  ·  {v}")
    axes[0][0].set_ylabel("Background (q/g) pass rate")
    axes[0][0].legend(loc="lower right", fontsize=8)
    save(fig, "paper2_roc")

    res = [("dsp", "DSPs"), ("ff", "Flip-flops"), ("lut", "LUTs")]
    fig, axes = plt.subplots(len(versions), 3, figsize=(14, 3.6 * len(versions)), squeeze=False)
    for row, v in zip(axes, versions):  # paper Figs. 11-12: resources vs precision per reuse factor
        for ax, (col, label) in zip(row, res):
            for rf in rfs:
                ax.plot(x_bits, scan(v, col, rf), marker="o", color=rf_color.get(rf, ink), label=f"reuse {rf}")
            ax.axhline(Z7020[col], color=ink2, lw=1, ls=":")
            ax.set_xticks(x_bits, ticks)
            ax.set(ylabel=label, yscale="symlog" if col == "dsp" else "log")
            if col == "dsp":
                ax.set_ylim(bottom=0)  # symlog would otherwise show negative DSP counts
            corner(ax, f"hls4ml  ·  {v}")
        row[0].legend(loc="lower right")
    save(fig, "paper3_resources")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for v, ls in zip(versions, ("-", "--", ":", "-.")):  # paper Fig. 14: latency and initiation interval
        for rf in rfs:
            axes[0].plot(x_bits, scan(v, "latency_ns", rf), marker="o", color=rf_color.get(rf, ink), ls=ls,
                         label=f"{v}, reuse {rf}")
            axes[1].plot(x_bits, CLOCK_NS * scan(v, "ii", rf), marker="o", color=rf_color.get(rf, ink), ls=ls)
    axes[1].axhline(25, color=ink2, lw=1, ls=":")
    axes[0].set(ylabel="Latency (ns)")
    axes[1].set(ylabel="Initiation interval (ns)")
    for ax in axes:
        ax.set_xticks(x_bits, ticks)
        ax.set(xlabel="Fixed-point precision", ylim=(0, None))
        corner(ax, "hls4ml  ·  xc7z020, 200 MHz")
    axes[0].legend(ncol=2, fontsize=8)
    save(fig, "paper4_latency")

    for v in versions:  # v4's self-correcting trigger: rate held on budget as each drift grows
        rows = self_correcting(ok[ok["version"] == v])
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.2))
        x = np.arange(len(rows))
        for s in SIGNALS:
            ax.plot(x, [r["rate"]["fixed"][s] for r in rows], marker="o", color=tagger[s], ls="--", label=f"{s}, fixed")
            ax.plot(x, [r["rate"]["adapt"][s] for r in rows], marker="o", color=tagger[s], label=f"{s}, self-correcting")
        ax.axhline(1, color=ink2, lw=1, ls=":")
        ax.set_xticks(x, [DRIFT_LABEL[k].format(l) for k, l in DRIFTS[1:]], rotation=30, ha="right")
        ax.set(ylabel="Background rate ÷ budget")
        ax.legend(ncol=3, fontsize=8)
        corner(ax, f"hls4ml  ·  {v}  ·  self-correcting trigger")
        save(fig, f"paper5_self_correcting_{v}")


if __name__ == "__main__":
    picked = [a for a in sys.argv[1:] if not a.startswith("--")]
    focus = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tab=")), None)
    df = load_results()
    if picked:
        df = df[df["version"].isin(picked)].reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    tmp = OUT / "data.js.tmp"  # written then renamed, so a page refreshing mid-write never reads half a file
    tmp.write_text("window.SWEEP = " + json.dumps(payload(df, "--live" in sys.argv, focus)) + ";\n")
    os.replace(tmp, OUT / "data.js")
    shutil.copyfile(PAGE, OUT / "report.html")
    print(f"{OUT / 'report.html'}: {len(df)} designs, versions {', '.join(sorted(df['version'].unique()))}")
    if "--png" in sys.argv:
        write_pngs(df, OUT / "png")
        print(f"paper-style images: {OUT / 'png'}/")
    if "--open" in sys.argv:  # WSL: hand the page to Windows, which opens it in the default browser
        win = subprocess.run(["wslpath", "-w", str((OUT / "report.html").resolve())], capture_output=True, text=True)
        subprocess.run(["explorer.exe", win.stdout.strip()])

"""Numbers from the sweep: one table of every design (results/results.csv), each version compared with the one
before on the same designs (results/vX_vs_vY.csv), and a short printed summary. The graphs are made by plots.py."""
import glob
import json
import textwrap

import numpy as np
import pandas as pd

from drift import DRIFTS
from trigger import SIGNALS

Z7020 = dict(lut=53_200, ff=106_400, dsp=220, bram18=280)  # xc7z020: the chip the designs are synthesised for
VU9P = dict(lut=1_182_240, dsp=6_840)                        # XCVU9P: L1-trigger-class FPGA, shown for context
MIN_ACC = 0.70  # designs below this have collapsed (over-pruned); left out of the summary plots, listed in the printout
DRIFT_COLS = [f"drift_{k}_{l}" for k, l in DRIFTS[1:]]
CLOCK_NS = 5.0


def load_results():
    df = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob("results/points/*.json"))])
    for k, n in Z7020.items():
        df[f"{k}_pct_z7020"] = 100 * df[k] / n
    df["fits_z7020"] = np.all([df[k] <= n for k, n in Z7020.items()], axis=0)
    for k, n in VU9P.items():
        df[f"{k}_pct_vu9p"] = 100 * df[k] / n
    df["mean_drift_drop"] = 100 * (df["acc_hls"].to_numpy()[:, None] - df[DRIFT_COLS].to_numpy()).mean(axis=1)
    for s in SIGNALS:  # background passing at the clean threshold under a 5% mass drift, relative to the clean rate
        df[f"rate_x_{s}_mass5"] = df[f"bkg_{s}_mass_scale_0.05"] / df[f"bkg_{s}"].where(df[f"bkg_{s}"] > 0)
    return df


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def show(d, cols):
    """Print selected columns with readable headers and formats: cols = [(column, header, format), ...]."""
    print(pd.DataFrame({h: d[c].map(f.format) for c, h, f in cols}).to_string(index=False))


if __name__ == "__main__":
    df = load_results()
    df.drop(columns=[c for c in df.columns if c.startswith(("roc_", "trace_")) or c == "thresholds"]).to_csv(
        "results/results.csv", index=False)
    versions = sorted(df["version"].unique())
    ok = df[df["acc_hls"] >= MIN_ACC]

    section("Designs per version")
    per = pd.DataFrame({
        "designs": df.groupby("version").size(),
        "fit xc7z020": df.groupby("version")["fits_z7020"].sum(),
        "collapsed": df[df["acc_hls"] < MIN_ACC].groupby("version").size(),
        "best accuracy": 100 * df.groupby("version")["acc_hls"].max(),
        "best that fits": 100 * df[df["fits_z7020"]].groupby("version")["acc_hls"].max(),
        "avg drift loss": ok.groupby("version")["mean_drift_drop"].mean(),
        "avg top eff @1%": 100 * ok.groupby("version")["eff1_t"].mean(),
    }).fillna(0)
    print(per.to_string(formatters={"best accuracy": "{:.2f}%".format, "best that fits": "{:.2f}%".format,
                                    "avg drift loss": "{:.2f} pts".format, "avg top eff @1%": "{:.1f}%".format,
                                    "collapsed": "{:.0f}".format}))

    # Each version against the one before, for the same design (bits, size, pruning, reuse factor)
    key = ["bits_label", "scale", "sparsity", "rf"]
    cols = ["acc_hls", "mean_drift_drop", "eff1_t", "eff1_w", "eff1_z", "rate_x_t_mass5", "rate_x_w_mass5", "n_inputs",
            "dsp", "lut", "ii"]
    labels = {"acc_hls": ("Accuracy", 100, "pts"), "mean_drift_drop": ("Accuracy lost under drift", 1, "pts"),
              "eff1_t": ("Top efficiency @1% bkg", 100, "pts"), "eff1_w": ("W efficiency @1% bkg", 100, "pts"),
              "eff1_z": ("Z efficiency @1% bkg", 100, "pts"), "n_inputs": ("Inputs", 1, ""),
              "dsp": ("DSPs", 1, ""), "lut": ("LUTs", 1, ""), "ii": ("New jet every (cycles)", 1, "")}
    changes = {}
    for a, b in zip(versions, versions[1:]):
        both = df[df["version"] == a][key + cols].merge(df[df["version"] == b][key + cols], on=key,
                                                        suffixes=(f"_{a}", f"_{b}"))
        for c in cols:
            both[f"d_{c}"] = both[f"{c}_{b}"] - both[f"{c}_{a}"]
        both.to_csv(f"results/{a}_vs_{b}.csv", index=False)
        if len(both):
            changes[f"{a} -> {b} ({len(both)} designs)"] = {
                lab: f"{scale * both[f'd_{c}'].mean():+,.2f} {unit}".strip() for c, (lab, scale, unit) in labels.items()}
    if changes:
        section("Each version vs the one before (same designs, average change)")
        print(pd.DataFrame(changes).to_string())

    section("Best 3 designs per version that fit the xc7z020")
    best = df[df["fits_z7020"]].sort_values("acc_hls", ascending=False).groupby("version").head(3).sort_values(
        ["version", "acc_hls"], ascending=[True, False]).assign(every=lambda d: d["ii"] * CLOCK_NS)
    show(best, [("name", "design", "{}"), ("acc_hls", "accuracy", "{:.2%}"), ("eff1_t", "top eff", "{:.1%}"),
                ("mean_drift_drop", "drift loss", "{:.1f} pts"), ("latency_ns", "latency", "{:.0f} ns"),
                ("every", "new jet every", "{:.0f} ns"), ("dsp", "DSPs", "{:d}"), ("lut", "LUTs", "{:,d}")])

    if any(c.startswith("bkg_adapt_") for c in df.columns):
        section("Self-correcting trigger (v4): background rate as a multiple of the budget, and signal efficiency")
        sc = ok[ok.filter(like="bkg_adapt_").notna().any(axis=1)]
        rows = []
        for kind, level in DRIFTS[1:]:
            k = f"{kind}_{level}"
            row = {"drift": k.replace("_scale", "")}
            for s in SIGNALS:  # each design's rate relative to its own budget (its rate before the drift)
                row[f"{s} fixed"] = (sc[f"bkg_fixed_{s}_{k}"] / sc[f"bkg_budget_{s}_{k}"]).mean()
                row[f"{s} self"] = (sc[f"bkg_adapt_{s}_{k}"] / sc[f"bkg_budget_{s}_{k}"]).mean()
            row["eff fixed"] = 100 * np.mean([sc[f"eff_fixed_{s}_{k}"].mean() for s in SIGNALS])
            row["eff self"] = 100 * np.mean([sc[f"eff_adapt_{s}_{k}"].mean() for s in SIGNALS])
            row["eff best"] = 100 * np.mean([sc[f"eff_oracle_{s}_{k}"].mean() for s in SIGNALS])
            rows.append(row)
        t = pd.DataFrame(rows)
        print(t.to_string(index=False, formatters={c: ("{:.2f}x".format if c.endswith(("fixed", "self")) and
                                                        not c.startswith("eff") else "{:.1f}%".format)
                                                   for c in t.columns if c != "drift"}))
        print(f"(average over {len(sc)} designs; 1.00x = on budget; eff = signal kept, averaged over t/w/z; "
              "best = fresh labelled recalibration)")

    section(f"Collapsed designs (accuracy < {MIN_ACC:.0%}, left out of the plots)")
    names = df.loc[df["acc_hls"] < MIN_ACC, "name"].tolist()
    print(textwrap.fill(", ".join(names), 110) if names else "none")
    print("\nEvery design: results/results.csv    Graphs: python plots.py --open")

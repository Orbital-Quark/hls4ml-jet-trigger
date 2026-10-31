"""Train, convert and synthesise every design in the grid, then evaluate it on the circuit emulation.
  VERSIONS=v4 N_JOBS=4 python sweep.py
The terminal shows one line per finished design; everything else (TensorFlow, training epochs, Vitis) goes to
logs/. Each version being run gets its own graphs (figures/<version>/), redrawn every few designs; the gallery
page opens once the first graphs exist and updates itself. At the end the summary (analyze.py) is printed and
the final graphs are drawn.
LIVE=0 turns the live graphs off; OPEN=0 stops the browser from opening."""
import json
import multiprocessing as mp
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # parallel jobs share the GPU instead of each grabbing all of it
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

POINTS, FAILED, LOGS = Path("results/points"), Path("results/failed"), Path("logs")
PART = os.environ.get("FPGA_PART", "xc7z020clg400-1")
N_EVAL = 50000          # test jets (and validation jets, for thresholds) run through the circuit emulation
REUSE = tuple(int(r) for r in os.environ.get("REUSE", "1,4").split(","))
MIXED = ((8, 4, 4, 8), (8, 6, 6, 8))   # v2+ per-layer bit widths (fc1, fc2, fc3, output)
LIVE_EVERY = 6          # redraw the graphs after this many finished designs


def version_flags(v):
    from pipeline import V1, V2, V3, V4
    return {"v1": V1, "v2": V2, "v3": V3, "v4": V4}[v]


def model_configs():
    versions = os.environ.get("VERSIONS", "v1,v2").split(",")
    if os.environ.get("GRID") == "test":  # two tiny models per version, to check the whole chain quickly
        grid = [(4, 0.25, 0.0), ((8, 4, 4, 8), 0.25, 0.5)]
    elif os.environ.get("GRID") == "smoke":  # the two CP1 smoke-test models, plus one pruned low-bit model
        grid = [(None, 0.25, 0.0), (None, 1.0, 0.0), (4, 0.25, 0.5)]
    else:
        grid = [(None, 1.0, 0.0)] + [(b, s, p) for b in (16, 8, 6, 4) for s in (1.0, 0.5, 0.25) for p in (0.0, 0.5, 0.75)]
    cfgs = []
    for v in versions:
        extra = [(m, 1.0, p) for m in MIXED for p in (0.0, 0.5)] if v != "v1" and not os.environ.get("GRID") else []
        for b, s, p in grid + extra:
            if v == "v1" and isinstance(b, tuple):
                continue
            cfgs.append(dict(version=v, bits=b, scale=s, sparsity=p))
    cfgs.sort(key=lambda c: c["scale"])  # small (fast) models first
    shard, n_shards = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
    return [c for i, c in enumerate(cfgs) if i % n_shards == shard]


def bits_label(b):
    return "float" if b is None else "-".join(map(str, b)) if isinstance(b, (tuple, list)) else str(b)


def point_name(c, rf):
    return f"{c['version']}_b{bits_label(c['bits'])}_s{int(c['scale'] * 100)}_p{int(c['sparsity'] * 100)}_r{rf}"


def quiet(log):
    """Send everything this worker process prints (TensorFlow, training epochs, Vitis) to a log file."""
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "a", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)


def summary(row):
    """One readable line of the numbers that matter for a finished design."""
    from analyze import DRIFT_COLS, Z7020
    drop = 100 * (row["acc_hls"] - sum(row[c] for c in DRIFT_COLS) / len(DRIFT_COLS))
    fits = all(row[k] <= n for k, n in Z7020.items())
    return (f"acc {row['acc_hls']:6.2%}  top@1% {row['eff1_t']:5.1%}  drift {-drop:5.1f}  DSP {row['dsp']:>5}  "
            f"LUT {row['lut']:>7,}  {row['latency_ns']:>4.0f} ns  {'fits chip' if fits else 'too big'}")


def run_model(c):
    todo = [rf for rf in REUSE if not (POINTS / f"{point_name(c, rf)}.json").exists()]
    if not todo:
        return [dict(status="skip", name=point_name(c, rf)) for rf in REUSE]
    quiet(LOGS / "points" / f"{point_name(c, todo[0])[:-3]}.log")
    try:
        import numpy as np

        from common import classes
        from convert import convert_and_synth, read_report
        from drift import DRIFTS, apply_drift
        from pipeline import build_data
        from train import train
        from trigger import SelfCorrectingMenu, TriggerMenu, roc

        d = build_data(version_flags(c["version"]))
        model = train(c, d.X_fit, d.y_fit, d.X_val, d.y_val, targets=d.targets, weights=d.weights)  # once per config
        cl = classes()
        Xe_raw, ye = d.X_te_raw[:N_EVAL], d.y_te[:N_EVAL]
        Xe = np.ascontiguousarray(d.prep.transform(Xe_raw))
        Xv, yv = np.ascontiguousarray(d.X_val[:N_EVAL]), d.y_val[:N_EVAL]
        acc_keras = float((model.predict(Xe, verbose=0).argmax(1) == ye).mean())
        nonzero = int(sum((l.get_weights()[0] != 0).sum() for l in model.layers if l.get_weights()))
    except Exception:
        FAILED.mkdir(parents=True, exist_ok=True)
        (FAILED / f"{point_name(c, todo[0])}_train.txt").write_text(traceback.format_exc())
        return [dict(status="FAIL", name=point_name(c, rf), summary="training failed") for rf in todo]

    out = []
    for rf in todo:
        name = point_name(c, rf)
        try:
            hls_model = convert_and_synth(model, name, PART, rf)
            row = dict(name=name, **c, bits_label=bits_label(c["bits"]), rf=rf, part=PART,
                       n_inputs=len(d.prep.names), acc_keras=acc_keras, nonzero_weights=nonzero,
                       rf_layers=hls_model.reuse_layers, **read_report(f"hls_prjs/{name}"))
            logits = hls_model.predict(Xe)                     # the circuit's 5 raw scores per jet
            row["acc_hls"] = float((logits.argmax(1) == ye).mean())
            adaptive = version_flags(c["version"]).get("adaptive")
            Menu = SelfCorrectingMenu if adaptive else TriggerMenu
            menu = Menu(cl).calibrate(hls_model.predict(Xv), yv)  # thresholds from validation jets
            row["thresholds"] = menu.thresholds
            row.update(roc(logits, ye, cl))
            row.update(menu.evaluate(logits, ye))
            for kind, level in DRIFTS[1:]:
                rng = np.random.default_rng(1234)  # same drifted data for every model
                Xd = np.ascontiguousarray(d.prep.transform(apply_drift(Xe_raw, kind, level, d.std, rng, d.names)))
                ld = hls_model.predict(Xd)
                row[f"drift_{kind}_{level}"] = float((ld.argmax(1) == ye).mean())
                for k, v in menu.evaluate(ld, ye).items():   # same thresholds: how efficiency and rate move
                    row[f"{k}_{kind}_{level}"] = v
                if adaptive:  # v4: let each trigger correct its own threshold on a drifted stream
                    sc = menu.self_correct(logits, ld, ye, np.random.default_rng(99),
                                           trace=(kind, level) == ("mass_scale", 0.1))
                    row.update({f"{k}_{kind}_{level}": v for k, v in sc.items()})
            POINTS.mkdir(parents=True, exist_ok=True)
            (POINTS / f"{name}.json").write_text(json.dumps(row, indent=1))
            out.append(dict(status="ok", name=name, summary=summary(row)))
        except Exception:
            FAILED.mkdir(parents=True, exist_ok=True)
            (FAILED / f"{name}.txt").write_text(traceback.format_exc())
            out.append(dict(status="FAIL", name=name, summary=f"see {FAILED / name}.txt"))
    return out


def prepare(versions):
    quiet(LOGS / "prepare.log")
    from pipeline import ensure_selection, ensure_teacher
    out = [ensure_teacher(version_flags(v)) for v in versions if version_flags(v).get("distill")]
    return out + [ensure_selection(version_flags(v)) for v in versions if version_flags(v).get("select")]


def clock(s):
    return time.strftime("%H:%M:%S", time.gmtime(s))


class LivePlots:
    """Redraw each running version's graphs in the background every few designs; open its gallery the first time."""

    def __init__(self, versions):
        self.versions, self.proc, self.opened = versions, None, False
        self.on = os.environ.get("LIVE", "1") != "0"

    def update(self, final=False):
        if not (self.on or final) or (self.proc and self.proc.poll() is None and not final):
            return
        if self.proc:
            self.proc.wait()
        flags = [] if final else ["--live"]
        if not self.opened and os.environ.get("OPEN", "1") != "0":
            flags.append("--open")
            self.opened = True
        chain = "; ".join(shlex.join([sys.executable, "plots.py", v, *flags]) for v in self.versions)
        log = open(LOGS / "plots.log", "a")
        self.proc = subprocess.Popen(["sh", "-c", chain], stdout=log, stderr=log)
        if final:
            self.proc.wait()


if __name__ == "__main__":
    LOGS.mkdir(exist_ok=True)
    cfgs = model_configs()
    versions = sorted({c["version"] for c in cfgs})
    total = len(cfgs) * len(REUSE)
    print(f"Sweep {', '.join(versions)}: {len(cfgs)} models x reuse {REUSE} = {total} designs, "
          f"{os.environ.get('N_JOBS', 2)} at a time")
    print(f"Detailed logs: {LOGS}/   Graphs: {', '.join(f'figures/{v}/index.html' for v in versions)} "
          "(update as designs finish)\n", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:  # teachers and feature selections first, in their own process so the GPU is freed after
        for line in pool.apply(prepare, (versions,)):
            print(f"  {line}", flush=True)

    live, start, done, new, skipped, failed = LivePlots(versions), time.time(), 0, 0, 0, 0
    with ctx.Pool(int(os.environ.get("N_JOBS", 2)), maxtasksperchild=1) as pool:
        for results in pool.imap_unordered(run_model, cfgs):
            for r in results:
                done += 1
                if r["status"] == "skip":
                    skipped += 1
                    continue
                new += 1
                failed += r["status"] == "FAIL"
                el = time.time() - start
                eta = el / new * (total - done)
                print(f"[{done:>3}/{total}] {clock(el)} elapsed, ~{clock(eta)} left | {r['status']:<4} "
                      f"{r['name']:<24} {r.get('summary', '')}", flush=True)
            if new and new % LIVE_EVERY == 0:
                live.update()

    print(f"\nDone in {clock(time.time() - start)}: {new - failed} new designs, {skipped} already done, "
          f"{failed} failed{' (see results/failed/)' if failed else ''}.", flush=True)
    subprocess.run([sys.executable, "analyze.py"])
    live.update(final=True)
    print(f"Final graphs: {', '.join(f'figures/{v}/' for v in versions)}  "
          "(view again: python plots.py <version> --open; compare: python plots.py v1 v2 v4)")

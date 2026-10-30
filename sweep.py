import json
import multiprocessing as mp
import os
import traceback
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # parallel jobs share the GPU instead of each grabbing all of it

POINTS, FAILED = Path("results/points"), Path("results/failed")
PART = os.environ.get("FPGA_PART", "xc7z020clg400-1")
N_EVAL = 50000          # test jets (and validation jets, for thresholds) run through the circuit emulation
REUSE = tuple(int(r) for r in os.environ.get("REUSE", "1,4").split(","))
MIXED = ((8, 4, 4, 8), (8, 6, 6, 8))   # v2 per-layer bit widths (fc1, fc2, fc3, output)


def version_flags(v):
    from pipeline import V1, V2
    return {"v1": V1, "v2": V2}[v]


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
        extra = [(m, 1.0, p) for m in MIXED for p in (0.0, 0.5)] if v == "v2" and not os.environ.get("GRID") else []
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


def run_model(c):
    todo = [rf for rf in REUSE if not (POINTS / f"{point_name(c, rf)}.json").exists()]
    if not todo:
        return [f"skip {point_name(c, rf)}" for rf in REUSE]
    try:
        import numpy as np

        from common import classes
        from convert import convert_and_synth, read_report
        from drift import DRIFTS, apply_drift
        from pipeline import build_data
        from train import train
        from trigger import TriggerMenu, roc

        d = build_data(version_flags(c["version"]))
        model = train(c, d.X_fit, d.y_fit, d.X_val, d.y_val, targets=d.targets)  # train once per model config
        cl = classes()
        Xe_raw, ye = d.X_te_raw[:N_EVAL], d.y_te[:N_EVAL]
        Xe = np.ascontiguousarray(d.prep.transform(Xe_raw))
        Xv, yv = np.ascontiguousarray(d.X_val[:N_EVAL]), d.y_val[:N_EVAL]
        acc_keras = float((model.predict(Xe, verbose=0).argmax(1) == ye).mean())
        nonzero = int(sum((l.get_weights()[0] != 0).sum() for l in model.layers if l.get_weights()))
    except Exception:
        FAILED.mkdir(parents=True, exist_ok=True)
        (FAILED / f"{point_name(c, todo[0])}_train.txt").write_text(traceback.format_exc())
        return [f"FAIL {point_name(c, rf)} (training)" for rf in todo]

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
            menu = TriggerMenu(cl).calibrate(hls_model.predict(Xv), yv)  # thresholds from validation jets
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
            POINTS.mkdir(parents=True, exist_ok=True)
            (POINTS / f"{name}.json").write_text(json.dumps(row, indent=1))
            out.append(f"ok   {name}")
        except Exception:
            FAILED.mkdir(parents=True, exist_ok=True)
            (FAILED / f"{name}.txt").write_text(traceback.format_exc())
            out.append(f"FAIL {name}")
    return out


def prepare_teachers(versions):
    from pipeline import ensure_teacher
    return [ensure_teacher(version_flags(v)) for v in versions if version_flags(v).get("distill")]


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:  # distillation teachers first, in their own process so the GPU is freed after
        for line in pool.apply(prepare_teachers, (sorted({c["version"] for c in model_configs()}),)):
            print(line, flush=True)
    n_jobs = int(os.environ.get("N_JOBS", 2))
    with ctx.Pool(n_jobs, maxtasksperchild=1) as pool:
        for lines in pool.imap_unordered(run_model, model_configs()):
            for line in lines:
                print(line, flush=True)

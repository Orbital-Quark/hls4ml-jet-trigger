"""v5.2 = v5 + physics-based input self-calibration (features.InputCalibrator), on v5's own trained networks.
No retraining and no new synthesis: each v5 network is reloaded from its build folder, converted and compiled for
the bit-accurate circuit emulation, and tested on every drift with the calibration switched on. Chip resources are
v5's; the calibration block itself (two counters, two registers, a multiplier and a subtractor) is not synthesised.
  N_JOBS=4 python selfcal.py          CHECK=1: no calibration, to confirm the reloaded networks reproduce v5"""
import json
import multiprocessing as mp
import os
import shutil
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sweep import FAILED, LOGS, PART, POINTS, clock, quiet, summary, version_flags

SRC, DST = "v5", "v5.2"
CHECK = os.environ.get("CHECK") == "1"
TMP = Path("hls_tmp")  # emulation-only builds, deleted after each design


def run(path):
    src = json.loads(Path(path).read_text())
    name = DST + src["name"][len(SRC):]
    if not CHECK and (POINTS / f"{name}.json").exists():
        return dict(status="skip", name=name)
    quiet(LOGS / "points" / f"{name}.log")
    try:
        import keras
        from qkeras.utils import _add_supported_quantized_objects

        from convert import convert_and_synth
        from features import InputCalibrator
        from pipeline import build_data
        from sweep import evaluate

        objects = {}
        _add_supported_quantized_objects(objects)
        model = keras.models.load_model(f"hls_prjs/{src['name']}/keras_model.keras", custom_objects=objects, compile=False)
        d = build_data(version_flags(DST), with_targets=False)
        hls_model = convert_and_synth(model, name, PART, src["rf"], root=str(TMP), synth=False)
        calib = None if CHECK else InputCalibrator(d.names).fit(d.X_val_raw)
        row = dict(src, name=name, version=DST, **evaluate(hls_model, d, True, calib))
        shutil.rmtree(TMP / name, ignore_errors=True)
        if CHECK:
            diff = max(abs(row[k] - src[k]) for k in src if k.startswith(("acc_hls", "drift_", "unseen_")))
            return dict(status="ok", name=src["name"], summary=f"largest difference from the v5 sweep: {diff:.6f}")
        row["calibration_ref"] = calib.ref.tolist()  # healthy medians of jet mass and particle count
        (POINTS / f"{name}.json").write_text(json.dumps(row, indent=1))
        return dict(status="ok", name=name, summary=summary(row))
    except Exception:
        FAILED.mkdir(parents=True, exist_ok=True)
        (FAILED / f"{name}.txt").write_text(traceback.format_exc())
        return dict(status="FAIL", name=name, summary=f"see {FAILED / name}.txt")


if __name__ == "__main__":
    paths = sorted(str(p) for p in POINTS.glob(f"{SRC}_*.json"))
    if CHECK:
        paths = paths[:2]
    print(f"{DST} from {len(paths)} {SRC} networks{' (check: no calibration)' if CHECK else ''}", flush=True)
    start = time.time()
    with mp.get_context("spawn").Pool(int(os.environ.get("N_JOBS", 4)), maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run, paths), 1):
            print(f"[{i:>3}/{len(paths)}] {clock(time.time() - start)} | {r['status']:<4} {r['name']:<26} {r.get('summary', '')}",
                  flush=True)

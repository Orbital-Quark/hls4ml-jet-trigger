# hls4ml jet trigger

A neural-network jet tagger (quark, gluon, W, Z, top) built for a small FPGA (xc7z020) with hls4ml. It follows the
hls4ml paper (arXiv:1804.06913), then improves on it: robustness to detector drift, cheaper designs, and a trigger
that keeps its own rate on budget.

| Version | What it adds |
|---|---|
| v1 | The paper's setup, reproduced |
| v2 | Drift-aware training, input clean-up, distillation from a float teacher, mixed per-layer precision |
| v3 | Rate weights, flat-mass weights, feature selection (tested in the ablation; only rate weights helped) |
| v4 | v2 + rate weights + a self-correcting trigger threshold |
| v5 | v4 with softer rate weights, 30% drifted training jets, and a 16th (always-zero) input so the chip takes a new jet every 20 ns |
| v5.2 | v5's networks + physics-based input self-calibration: the stream's median jet mass and particle count undo energy-scale slips and pileup |

**Setup and running:** see [SETUP.md](SETUP.md).

```bash
python prepare_data.py                     # download the dataset once
VERSIONS=v5 N_JOBS=4 python sweep.py       # train, synthesise and test every design
N_JOBS=4 python selfcal.py                 # v5.2: v5's networks retested with the input self-calibration
python plots.py --open                     # interactive results page
```

| File | Does |
|---|---|
| `sweep.py` | Runs the whole grid: train → convert → synthesise → test on the circuit emulation |
| `pipeline.py` | What each version changes (flags), data preparation, teacher models |
| `models.py`, `train.py` | Network, pruning, training (incl. the distillation loss) |
| `features.py`, `weights.py`, `drift.py` | Input scaling, training weights, detector-drift simulation |
| `convert.py` | hls4ml conversion and Vitis HLS synthesis |
| `trigger.py` | Trigger thresholds at 1% background, ROC/AUC, the self-correcting trigger |
| `analyze.py`, `plots.py`, `report.html` | Summary tables and the results page |
| `ablation.py` | Which single change helps (Keras only) |
| `selfcal.py` | v5.2: reloads the v5 networks and tests them with the input self-calibration (`features.InputCalibrator`) |

## 2026-10-20 — A
- Did: Set up WSL Ubuntu 22.04, conda env (hls4ml-tutorial), started Vitis 2024.1 install (Zynq-7000 only), created repo
- Broke: First paste into WSL added ^[[200~ junk — fixed by opening Ubuntu directly
- Broke: Vitis install crashed at 96% — C: filled up because the WSL disk file grows but never shrinks (hit 120 GB). Fixed by freeing space, reinstalling from cached downloads, then fstrim + diskpart compact (120 → 79 GB).
- Decided / rejected: Ubuntu 22.04 over 26.04 (Vitis needs libtinfo5)

## 2026-10-25 — A
- Did: Wrote and ran prepare_data.py (5.1). 830k jets × 16 features, 5 classes, 80/20 stratified split. Sanity-checked: no NaN/inf, classes ~20% each in train and test, scaler gives mean 0 / std 1, median mmdt jet mass per class q 8.8, g 41.2, W 79.5, Z 90.3, t 153.7 GeV (matches the W/Z/top masses).
- Broke: First run used Windows Python 3.14 via VS Code's Run button → OpenML md5 checksum error. File itself was fine (md5 matched when downloaded in Linux). Fix: run only from the Ubuntu terminal with the conda env active.
- Decided / rejected: d2_b1_mmdt and d2_a1_b1_mmdt are exact duplicates in the dataset (15 distinct features, not 16). Kept all 16 to stay comparable with the hls4ml paper.

## 2026-10-26 — A
- Did: Wrote common.py (5.2: one shared loader for the data, feature names and scaler) and models.py (5.3: build_model(bits, scale) for float or QKeras QAT models, magnitude-pruning masks and the KeepPruned callback). Tested: the baseline 16→64→32→32→5 has 4,389 parameters (matches the hls4ml paper), the quantised layers are QDense/QActivation, and a 50% mask zeros exactly half of fc1's weights.
- Decided / rejected: Store the scaler with pickle. It's 32 numbers and must be reused unchanged, so drifted test data gets scaled with the training statistics. joblib would add nothing at this size, and the big arrays already use np.save. QAT (train with quantisation) rather than quantising after training as the 1804 paper does. One-shot magnitude pruning at fixed sparsities rather than the paper's iterative L1 pruning, so every point in the sweep is reproducible.

## 2026-10-27 — A
- Did: Wrote train.py (5.4). Quick check: trained the float baseline (16→64→32→32→5) on its own: test accuracy 76.8% (train 76.8%, val 76.9%, val_loss 0.643), matching the hls4ml paper's ~75–76%. 519 steps/epoch, ~3 s/epoch.
- Decided / rejected: Checked training on its own before adding hls4ml (5.5), so a bug can only be in data, model or training. The baseline used all 40 epochs without early stopping firing, so raised the cap to 100 and let early stopping decide. Otherwise smaller/quantised models could be cut off early and look worse than they are.

## 2026-10-28 — A
- Did: Wrote convert.py and drift.py (5.5). Drift check on the test set: mass_scale moves the W median mass 79.5 → 81.1 / 83.4 / 87.4 GeV (+2/5/10%); pileup raises mean multiplicity 52 → 54 / 57 / 62; smear blurs the mass (W median 78.7 → 75.8) and leaves multiplicity alone. Started the Part 6 smoke test: smoke_small (16-8-8, float) reached 76.1% in Keras.
- Decided / rejected: Renamed the model's input layer "input" → "jet_features", because hls4ml reserves the name "input" and refuses to convert. Switched the firmware softmax to hls4ml's "argmax" implementation: with the default "stable" softmax, Vitis HLS 2024.1 spent 25+ minutes stuck scheduling the exp lookup-table setup. A trigger only needs the top class, so accuracy is unchanged and the hardware is cheaper.

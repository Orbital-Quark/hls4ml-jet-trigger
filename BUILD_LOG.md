## 2026-10-20 — A
- Did: Set up Ubuntu 22.04 on WSL and the Python environment (hls4ml-tutorial), started installing Vitis 2024.1 (only the Zynq-7000 chip family), and created the repo.
- Decided / rejected: Went with Ubuntu 22.04 instead of 26.04, because Vitis needs an older library that the newer Ubuntu doesn't have.

## 2026-10-25 — A
- Did: Wrote and ran prepare_data.py (step 5.1). It loads about 830,000 jets, each described by 16 numbers and labelled as one of 5 types, and splits them 80% for training and 20% for testing. Checked the data: nothing missing or broken, each type makes up about 20% of both sets, and after scaling every feature is centred on 0. The typical jet mass for each type also matches the real particles (W 79.5, Z 90.3, top 153.7 GeV), so the data looks right.
- Decided / rejected: Two of the 16 features turned out to be exact copies of each other, so there are really only 15. Kept all 16 for now so our results can be compared directly with the hls4ml paper.

## 2026-10-26 — A
- Did: Wrote common.py (step 5.2), one shared place that loads the data, feature names and scaler, and models.py (step 5.3), which builds the network at any size and precision and can prune it (switch off its weakest connections). Tested it: the standard model has 4,389 adjustable numbers, the same as the hls4ml paper, and pruning 50% switches off exactly half the connections in the first layer.
- Decided / rejected: Saved the scaler with pickle. It's only 32 numbers and has to be reused exactly as it is, so that drifted test data gets scaled the same way as the training data. joblib wouldn't add anything at this size, and the big arrays are already saved with numpy. Trained the models with low precision built in from the start, instead of cutting precision after training like the original paper did. Pruned in one go at fixed amounts instead of the paper's gradual method, so every result can be reproduced exactly.

## 2026-10-27 — A
- Did: Wrote train.py (step 5.4). As a quick check, trained the standard full-precision model on its own: it classifies 76.8% of the test jets correctly, in line with the hls4ml paper's 75–76%. One pass over the training data takes about 3 seconds.
- Decided / rejected: Tested training by itself before adding hls4ml (step 5.5), so if something went wrong we'd know it was in the data, the model or the training. The model used all 40 training rounds without ever stopping early, so raised the limit to 100 and let the early-stop rule decide when it's done. Otherwise smaller models could get cut off too soon and look worse than they really are.

## 2026-10-28 — A
- Did: Wrote convert.py, which turns a trained model into a chip design, and drift.py, which fakes detector problems (step 5.5). Checked that the fake problems behave as expected: shifting the mass scale moves the typical W mass from 79.5 to 81.1, 83.4 and 87.4 GeV (+2, 5 and 10%); adding pileup raises the average number of particles per jet from 52 to 54, 57 and 62; and smearing blurs the mass but leaves the particle count alone. Started the Part 6 smoke test: the small model got 76.1% right.
- Decided / rejected: Renamed the model's input layer from "input" to "jet_features", because hls4ml keeps the name "input" for itself and refuses to convert otherwise. Changed the chip's last step (softmax) to simply pick the highest score: the normal version left Vitis stuck for over 25 minutes. A trigger only needs the top guess, so accuracy stays the same and the chip gets cheaper.

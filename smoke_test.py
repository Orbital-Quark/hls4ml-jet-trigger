import numpy as np

from common import load
from convert import convert_and_synth, read_report
from train import train

X_tr_raw, X_te_raw, y_tr, y_te, scaler, names = load()
X_tr = scaler.transform(X_tr_raw).astype(np.float32)
Xe = np.ascontiguousarray(scaler.transform(X_te_raw[:20000]), dtype=np.float32)
ye = y_te[:20000]

for cfg, name in [(dict(bits=None, scale=0.25, sparsity=0.0), "smoke_small"),   # fast: proves the toolchain
                  (dict(bits=None, scale=1.0, sparsity=0.0), "smoke_baseline")]: # the reference model
    model = train(cfg, X_tr, y_tr)
    print(name, "keras acc:", (model.predict(Xe, verbose=0).argmax(1) == ye).mean())
    hls_model = convert_and_synth(model, name, "xc7z020clg400-1", rf=4)
    print(name, "hls acc:  ", (hls_model.predict(Xe).argmax(1) == ye).mean())
    print(name, read_report(f"hls_prjs/{name}"))
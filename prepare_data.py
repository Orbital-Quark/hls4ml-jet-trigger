import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

out = Path("data")
out.mkdir(exist_ok=True)

d = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=False)
X = d["data"].astype(np.float32)
le = LabelEncoder()
y = le.fit_transform(d["target"])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler().fit(X_tr)

np.save(out / "X_train_raw.npy", X_tr)
np.save(out / "X_test_raw.npy", X_te)
np.save(out / "y_train.npy", y_tr)
np.save(out / "y_test.npy", y_te)
np.save(out / "classes.npy", le.classes_)
np.save(out / "feature_names.npy", np.array(d["feature_names"]))
with open(out / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("X:", X.shape, "| classes:", le.classes_)
print("features:", d["feature_names"])
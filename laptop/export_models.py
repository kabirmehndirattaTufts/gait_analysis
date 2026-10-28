"""
Turn the trained scikit-learn models into one plain JSON file the board can run with numpy only.

    py export_models.py --models models

Reads models/*.joblib (made by train.py) and writes models/models.json. It then checks that the numpy
version gives exactly the same answers as scikit-learn on every training window.
"""
import argparse
import json
import os

import joblib
import numpy as np

from tinyml import build

NAMES = ["model1_tree", "model1_knn", "model2_tree", "model2_knn"]


def export_one(m):
    if hasattr(m, "tree_"):                                    # decision tree
        t = m.tree_
        return {"type": "tree", "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                "value": t.value[:, 0, :].tolist(), "classes": m.classes_.tolist()}
    scaler, knn = m[0], m[-1]                                  # pipeline: StandardScaler + kNN
    assert knn.weights == "uniform" and knn.p == 2 and knn.metric == "minkowski"
    return {"type": "knn", "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "X": knn._fit_X.tolist(), "y": np.asarray(knn._y).tolist(), "k": int(knn.n_neighbors)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    loaded = {n: joblib.load(os.path.join(a.models, n + ".joblib")) for n in NAMES}
    out = {"axis": loaded["model1_tree"]["axis"], "classes": loaded["model1_tree"]["classes"]}
    for n in NAMES:
        out[n] = export_one(loaded[n]["model"])

    # the training windows (un-standardized) for each feature set, recovered from the kNN models
    raw = {}
    for tag in ("model1", "model2"):
        knn = loaded[tag + "_knn"]["model"]
        raw[tag] = knn._fit_X_raw if hasattr(knn, "_fit_X_raw") else \
            knn[-1]._fit_X * knn[0].scale_ + knn[0].mean_
    for n in NAMES:
        X = raw[n[:6]]
        ok = np.array_equal(np.asarray(loaded[n]["model"].predict(X)), build(out[n]).predict(X))
        print(f"   {n}: numpy version identical to scikit-learn on {len(X)} windows: {'YES' if ok else 'NO  <-- problem'}")
        if not ok:
            raise SystemExit("export check failed")
    path = os.path.join(a.models, "models.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

"""
Score the four trained models on NEW recordings they have never seen (a true held-out test set).

    py evaluate.py --data C:\\Users\\13013\\Documents\\gait_data --models models

Record the test files with session names t1, t2, ... so that train.py ignores them, for example:
    rec "still,t1,20,10"      -> still_t1.csv
    rec "walking,t1,20,10"    -> walking_t1.csv
    rec "walking,t2,20,10"    -> walking_t2.csv   (a different pace)

Needs train.py in the same folder (it reuses the exact same windowing and features) and the models that
train.py saved. Writes test_confusion.png.
"""
import argparse
import glob
import os
import re

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from train import CHANNELS, CLASSES, S, W, model1_features, model2_features

NAMES = ["model1_tree", "model1_knn", "model2_tree", "model2_knn"]
TITLES = {"model1_tree": "Model 1 + tree", "model1_knn": "Model 1 + kNN",
          "model2_tree": "Model 2 + tree", "model2_knn": "Model 2 + kNN"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()

    models = {n: joblib.load(os.path.join(a.models, n + ".joblib")) for n in NAMES}
    axis = CHANNELS.index(models["model1_tree"]["axis"])

    X1, X2, y, rec = [], [], [], []
    for path in sorted(glob.glob(os.path.join(a.data, "*_t[0-9]*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = stem.split("_")[0]
        if label not in CLASSES or not re.match(r"^[a-z]+_t\d+(_\d+)?$", stem):
            continue
        d = np.genfromtxt(path, delimiter=",", names=True)
        X = np.column_stack([d[c] for c in CHANNELS])
        for i in range(0, len(X) - W + 1, S):
            win = X[i:i + W]
            X1.append(model1_features(win, axis)); X2.append(model2_features(win))
            y.append(CLASSES.index(label)); rec.append(stem)
    if not y:
        raise SystemExit(f"No *_t<N>.csv test files found in {a.data}")
    X1, X2, y, rec = np.array(X1), np.array(X2), np.array(y), np.array(rec)
    print(f"{len(y)} test windows from {len(set(rec))} recordings: " +
          ", ".join(f"{c} {(y == i).sum()}" for i, c in enumerate(CLASSES)))

    preds = {n: models[n]["model"].predict(X1 if n.startswith("model1") else X2) for n in NAMES}

    print("\nTest-set accuracy")
    for n in NAMES:
        print(f"   {TITLES[n]:16s} {(preds[n] == y).mean() * 100:5.1f}%")

    print("\nPer-class recall (how often each real activity is recognized)")
    print(f"   {'':16s} " + " ".join(f"{c:>8s}" for c in CLASSES))
    for n in NAMES:
        print(f"   {TITLES[n]:16s} " + " ".join(
            f"{(preds[n][y == i] == i).mean() * 100:7.0f}%" if (y == i).any() else f"{'-':>8s}" for i in range(4)))

    print("\nPer recording")
    print(f"   {'recording':16s} " + " ".join(f"{TITLES[n]:>15s}" for n in NAMES))
    for r in dict.fromkeys(rec):
        m = rec == r
        print(f"   {r:16s} " + " ".join(f"{(preds[n][m] == y[m]).mean() * 100:14.0f}%" for n in NAMES))

    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for axx, n in zip(axs.ravel(), NAMES):
        cm = confusion_matrix(y, preds[n], labels=range(4))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        axx.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        axx.set_xticks(range(4)); axx.set_xticklabels(CLASSES); axx.set_yticks(range(4)); axx.set_yticklabels(CLASSES)
        axx.set_xlabel("predicted"); axx.set_ylabel("actual")
        axx.set_title(f"{TITLES[n]}   {(preds[n] == y).mean() * 100:.1f}%")
        for i in range(4):
            for j in range(4):
                axx.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    fig.suptitle("Held-out test recordings (counts of 3 s windows)")
    fig.tight_layout(); fig.savefig("test_confusion.png", dpi=130)
    print("\nSaved test_confusion.png")


if __name__ == "__main__":
    main()

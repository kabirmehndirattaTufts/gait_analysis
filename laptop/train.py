"""
Train and compare the activity classifiers (no neural nets).

    py train.py --data C:\\Users\\13013\\Documents\\gait_data --out models

It reads every  <label>_s<N>[_k].csv  file in --data (still_s1.csv, walking_s1_4.csv, ...) and ignores the rest
(pilot, check, test, ignore_ files). Each file is cut into 3 s windows with 50% overlap, and four models are
compared:

                      ONE feature (model 1)              ALL channels (model 2)
    decision tree     peak |a - mean| on one axis         30 features from all 6 IMU channels
    nearest neighbor  same one feature                    same 30 features

Honest scoring: leave-one-RECORDING-out. Every window of one recording is held out together, the models train on
the other recordings, and the held-out recording is classified. Overlapping windows from the same recording are
never split between train and test. Holding out a whole recording also tests an unseen pace (for example the
slow walk is predicted by a model that only saw the medium and fast walks).

Needs:  py -m pip install numpy scikit-learn matplotlib
Writes: confusion_matrices.png, model1_feature.png, model2_pca.png, and the four fitted models in --out.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

FS = 100
WINDOW_S = 3.0
STEP_S = 1.5
CLASSES = ["still", "walking", "running", "biking"]
CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
PEAK_AXIS = "ax_g"                         # the one axis model 1 uses (see the axis comparison printed below)
K_NEIGHBORS = 5

W = int(WINDOW_S * FS)
S = int(STEP_S * FS)


# ---------------------------------------------------------------------------------------------
# data and features
# ---------------------------------------------------------------------------------------------
def load_recordings(folder):
    recs = []
    for path in sorted(glob.glob(os.path.join(folder, "*_s[0-9]*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        label = stem.split("_")[0]
        if label not in CLASSES or not re.match(r"^[a-z]+_s\d+(_\d+)?$", stem):
            continue
        d = np.genfromtxt(path, delimiter=",", names=True)
        recs.append((stem, label, np.column_stack([d[c] for c in CHANNELS])))
    return recs


def dom_freq(x):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= 0.5) & (f <= 10)
    return f[m][spec[m].argmax()]


def model1_features(win, axis):
    x = win[:, axis]
    return [np.abs(x - x.mean()).max()]


MODEL2_NAMES = [f"{c}_{s}" for c in CHANNELS for s in ("std", "peak", "range", "domfreq")] + \
               ["amag_mean", "amag_std", "amag_max", "gmag_mean", "gmag_std", "gmag_max"]


def model2_features(win):
    f = []
    for c in range(6):
        x = win[:, c]
        xc = x - x.mean()                      # mean removed: strap tilt must not decide the class
        f += [xc.std(), np.abs(xc).max(), np.ptp(x), dom_freq(x)]
    am = np.linalg.norm(win[:, :3], axis=1)
    gm = np.linalg.norm(win[:, 3:], axis=1)
    f += [am.mean(), am.std(), am.max(), gm.mean(), gm.std(), gm.max()]
    return f


def build(recs, axis):
    X1, X2, y, g = [], [], [], []
    for stem, label, X in recs:
        for i in range(0, len(X) - W + 1, S):
            win = X[i:i + W]
            X1.append(model1_features(win, axis))
            X2.append(model2_features(win))
            y.append(CLASSES.index(label))
            g.append(stem)
    return np.array(X1), np.array(X2), np.array(y), np.array(g)


# ---------------------------------------------------------------------------------------------
# models and scoring
# ---------------------------------------------------------------------------------------------
def make_tree(depth):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=3, random_state=0)


def make_knn():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_NEIGHBORS))


def loro_predict(make, X, y, g):
    pred = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = make()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def acc(y, p):
    return float((y == p).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="folder with the CSV files")
    ap.add_argument("--out", default="models", help="folder for the fitted models")
    ap.add_argument("--kmeans", action="store_true", help="also run the unlabeled k-means cluster check")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    recs = load_recordings(a.data)
    if not recs:
        raise SystemExit(f"No <label>_s<N>.csv files found in {a.data}")
    print(f"{len(recs)} recordings:")
    for stem, label, X in recs:
        print(f"   {stem:16s} {label:8s} {len(X) / FS:5.1f} s")

    axis = CHANNELS.index(PEAK_AXIS)
    X1, X2, y, g = build(recs, axis)
    print(f"\n{len(y)} windows of {WINDOW_S:g} s  ->  " + ", ".join(f"{c}: {(y == i).sum()}" for i, c in enumerate(CLASSES)))

    # which single axis is best for model 1? (for reference; PEAK_AXIS decides what is used)
    print("\nModel 1 axis comparison (depth-3 tree, leave-one-recording-out accuracy):")
    for ax_i in range(3):
        xa, _, ya, ga = build(recs, ax_i)
        print(f"   peak on {CHANNELS[ax_i]:5s}: {acc(ya, loro_predict(lambda: make_tree(3), xa, ya, ga)) * 100:5.1f}%"
              + ("   <- used" if ax_i == axis else ""))

    combos = {
        "Model 1 + tree":   (X1, lambda: make_tree(3)),
        "Model 1 + kNN":    (X1, make_knn),
        "Model 2 + tree":   (X2, lambda: make_tree(5)),
        "Model 2 + kNN":    (X2, make_knn),
    }
    preds = {}
    print("\nLeave-one-recording-out results")
    print(f"   {'':16s} {'accuracy':>9s} {'balanced':>9s}")
    for name, (X, mk) in combos.items():
        p = loro_predict(mk, X, y, g)
        preds[name] = p
        print(f"   {name:16s} {acc(y, p) * 100:8.1f}% {balanced_accuracy_score(y, p) * 100:8.1f}%")

    print("\nAccuracy for each held-out recording (which paces break which model?)")
    names = list(combos)
    print(f"   {'recording':16s} " + " ".join(f"{n:>15s}" for n in names))
    for stem in dict.fromkeys(g):
        m = g == stem
        print(f"   {stem:16s} " + " ".join(f"{acc(y[m], preds[n][m]) * 100:14.0f}%" for n in names))

    print("\nWhere the tree and the nearest-neighbor model disagree:")
    for tag in ("Model 1", "Model 2"):
        pt, pk = preds[f"{tag} + tree"], preds[f"{tag} + kNN"]
        d = pt != pk
        print(f"   {tag}: they differ on {d.mean() * 100:.1f}% of windows; "
              f"when they differ the tree is right {(pt[d] == y[d]).mean() * 100 if d.any() else 0:.0f}%, "
              f"kNN is right {(pk[d] == y[d]).mean() * 100 if d.any() else 0:.0f}%")

    # confusion matrices
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for axx, name in zip(axs.ravel(), names):
        cm = confusion_matrix(y, preds[name], labels=range(4))
        cmn = cm / cm.sum(1, keepdims=True)
        axx.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        axx.set_xticks(range(4)); axx.set_xticklabels(CLASSES); axx.set_yticks(range(4)); axx.set_yticklabels(CLASSES)
        axx.set_xlabel("predicted"); axx.set_ylabel("actual")
        axx.set_title(f"{name}   {acc(y, preds[name]) * 100:.1f}%")
        for i in range(4):
            for j in range(4):
                axx.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black")
    fig.suptitle("Leave-one-recording-out confusion matrices (counts of 3 s windows)")
    fig.tight_layout(); fig.savefig("confusion_matrices.png", dpi=130); plt.close(fig)

    # fit on everything
    t1 = make_tree(3).fit(X1, y)
    k1 = make_knn().fit(X1, y)
    t2 = make_tree(5).fit(X2, y)
    k2 = make_knn().fit(X2, y)
    print(f"\nModel 1 tree rules ({PEAK_AXIS} peak, fit on all data):")
    print(export_text(t1, feature_names=[f"peak_{PEAK_AXIS}"]))
    top = np.argsort(t2.feature_importances_)[::-1][:8]
    print("Model 2 tree: most used features:")
    for i in top:
        if t2.feature_importances_[i] > 0:
            print(f"   {MODEL2_NAMES[i]:16s} {t2.feature_importances_[i]:.2f}")

    # model 1 picture: the feature per class with the tree's thresholds
    fig, ax1 = plt.subplots(figsize=(11, 4.2))
    cols = ["#6b7280", "#1b9e77", "#d95f02", "#7570b3"]
    rng = np.random.default_rng(0)
    for i, c in enumerate(CLASSES):
        ax1.scatter(X1[y == i, 0], i + rng.uniform(-.22, .22, (y == i).sum()), s=16, color=cols[i], alpha=.7, label=c)
    for n, thr in enumerate(sorted(t1.tree_.threshold[t1.tree_.feature >= 0])):
        ax1.axvline(thr, color="black", ls="--", lw=1)
        ax1.text(thr, 3.62 - 0.22 * (n % 2), f"{thr:.2f}", ha="center", fontsize=8, bbox=dict(fc="white", ec="none", pad=1))
    ax1.set_yticks(range(4)); ax1.set_yticklabels(CLASSES)
    ax1.set_xlabel(f"model 1 feature: peak |{PEAK_AXIS} - window mean|"); ax1.set_xscale("symlog", linthresh=0.1); ax1.set_xlim(0, None); ax1.set_ylim(-0.5, 3.9)
    ax1.set_title("One feature, four classes. Dashed lines = the decision tree's cut points (fit on all data)", pad=14)
    fig.tight_layout(); fig.savefig("model1_feature.png", dpi=130); plt.close(fig)

    # model 2 picture: all features squeezed to 2D
    Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X2))
    fig, ax2 = plt.subplots(figsize=(7.5, 6))
    for i, c in enumerate(CLASSES):
        ax2.scatter(Z[y == i, 0], Z[y == i, 1], s=18, color=cols[i], alpha=.75, label=c)
    ax2.set_xlabel("PCA 1"); ax2.set_ylabel("PCA 2"); ax2.legend(); ax2.set_title("Model 2 features squeezed to 2D (each dot = one 3 s window)")
    fig.tight_layout(); fig.savefig("model2_pca.png", dpi=130); plt.close(fig)

    if a.kmeans:
        Xs = StandardScaler().fit_transform(X2)
        km = KMeans(4, n_init=10, random_state=0).fit(Xs)
        print("\nk-means (no labels used) on the model 2 features: rows = clusters, columns = true class")
        print("   " + " ".join(f"{c:>8s}" for c in CLASSES))
        for k in range(4):
            print("   " + " ".join(f"{((km.labels_ == k) & (y == i)).sum():8d}" for i in range(4)) + f"   cluster {k}")
        print(f"   adjusted Rand index vs the true classes: {adjusted_rand_score(y, km.labels_):.2f} (1.0 = perfect)")

    for name, m, feats in (("model1_tree", t1, [f"peak_{PEAK_AXIS}"]), ("model1_knn", k1, [f"peak_{PEAK_AXIS}"]),
                           ("model2_tree", t2, MODEL2_NAMES), ("model2_knn", k2, MODEL2_NAMES)):
        joblib.dump({"model": m, "features": feats, "classes": CLASSES, "axis": PEAK_AXIS,
                     "window_s": WINDOW_S, "fs": FS}, os.path.join(a.out, name + ".joblib"))
    print(f"\nSaved the four fitted models to {a.out}/ and three figures to the current folder.")


if __name__ == "__main__":
    main()

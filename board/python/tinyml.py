"""The four trained models, run with numpy only (no scikit-learn needed on the board).
export_models.py on the laptop writes models/models.json; this file rebuilds the same predictions from it."""
import numpy as np


class TreeModel:
    def __init__(self, d):
        self.left = np.array(d["left"])
        self.right = np.array(d["right"])
        self.feature = np.array(d["feature"])
        self.threshold = np.array(d["threshold"], dtype=float)
        self.value = np.array(d["value"], dtype=float)         # one row of class weights per node
        self.classes = np.array(d["classes"])

    def predict(self, X):
        out = []
        for x in np.asarray(X, dtype=float):
            n = 0
            while self.left[n] != -1:                          # -1 marks a leaf
                n = self.left[n] if x[self.feature[n]] <= self.threshold[n] else self.right[n]
            out.append(self.classes[int(np.argmax(self.value[n]))])
        return np.array(out)


class KnnModel:
    def __init__(self, d):
        self.mean = np.array(d["mean"], dtype=float)
        self.scale = np.array(d["scale"], dtype=float)
        self.X = np.array(d["X"], dtype=float)                 # training windows, already standardized
        self.y = np.array(d["y"])
        self.k = int(d["k"])
        self.n_classes = int(max(self.y)) + 1

    def predict(self, X):
        Z = (np.asarray(X, dtype=float) - self.mean) / self.scale
        out = []
        for z in Z:
            dist = ((self.X - z) ** 2).sum(axis=1)
            nearest = np.argsort(dist, kind="stable")[:self.k]
            out.append(int(np.bincount(self.y[nearest], minlength=self.n_classes).argmax()))
        return np.array(out)


def build(d):
    return TreeModel(d) if d["type"] == "tree" else KnnModel(d)

"""Windowing and features. This is the SAME code as in train.py, so the live app computes exactly what was trained on."""
import numpy as np

FS = 100
WINDOW_S = 3.0
STEP_S = 1.5
CLASSES = ["still", "walking", "running", "biking"]
CHANNELS = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]
W = int(WINDOW_S * FS)
S = int(STEP_S * FS)


def dom_freq(x):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    m = (f >= 0.5) & (f <= 10)
    return f[m][spec[m].argmax()]


def model1_features(win, axis):
    x = win[:, axis]
    return [np.abs(x - x.mean()).max()]


def model2_features(win):
    f = []
    for c in range(6):
        x = win[:, c]
        xc = x - x.mean()
        f += [xc.std(), np.abs(xc).max(), np.ptp(x), dom_freq(x)]
    am = np.linalg.norm(win[:, :3], axis=1)
    gm = np.linalg.norm(win[:, 3:], axis=1)
    f += [am.mean(), am.std(), am.max(), gm.mean(), gm.std(), gm.max()]
    return f

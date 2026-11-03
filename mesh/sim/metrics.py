import numpy as np


def jain(x):
    x = np.asarray(x, dtype=float)
    s = x.sum()
    return 1.0 if s == 0 else float(s * s / (len(x) * (x * x).sum()))


def p5(x):
    x = np.asarray(x, dtype=float)
    return 0.0 if x.size == 0 else float(np.percentile(x, 5))


def utility(x, eps):
    return float(np.log(np.asarray(x, dtype=float) + eps).sum())

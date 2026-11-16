"""Expanding-window walk-forward with a purge.

A training row at time t has a label that looks at prices up to t + H bars. So a row may only be trained on if
t + H*tf <= test_start. The old 80/20 split in ml.py skipped this, letting the last H training labels peek into the
validation window. Splitting on *timestamps* (not row numbers) also keeps pooled multi-symbol data honest.
"""
from __future__ import annotations

import numpy as np


def walk_forward(ts: np.ndarray, horizon: int, tf_ms: int, n_folds: int = 4, min_train_frac: float = 0.5):
    """Yield (fold, train_mask, test_mask) over rows identified by `ts` (epoch ms)."""
    ts = np.asarray(ts, dtype="int64")
    uts = np.unique(ts)
    n = len(uts)
    if n < n_folds * 4:
        raise ValueError(f"only {n} distinct timestamps; need at least {n_folds * 4} for {n_folds} folds")
    start = int(n * min_train_frac)
    block = max(1, (n - start) // n_folds)
    for j in range(n_folds):
        a = uts[start + j * block]
        b = uts[start + (j + 1) * block] if j < n_folds - 1 else uts[-1] + 1
        yield j, (ts + horizon * tf_ms <= a), ((ts >= a) & (ts < b))

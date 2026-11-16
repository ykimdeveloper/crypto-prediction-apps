"""Label = "did price move far enough, in one direction, to pay for trading it?"

    fwd_t   = ln(close[t+H] / close[t])                       (NaN if there is a hole in the next H bars)
    theta_t = cost_bps/1e4 + k * sigma_t * sqrt(H)            sigma_t = per-bar EWMA vol known at t
    y_t     = 1 if fwd > theta,  0 if fwd < -theta,  NaN (neutral, not trained on) otherwise

Neutral rows are still used when *trading* is simulated - the model doesn't get to skip them for free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import MIN_MS


def make_labels(d: pd.DataFrame, sigma: pd.Series, horizon: int, cost_bps: float, k: float, tf_ms: int = MIN_MS) -> pd.DataFrame:
    c = d["close"].astype(float)
    fwd = np.log(c.shift(-horizon) / c)
    fwd = fwd.where((d["ts"].shift(-horizon) - d["ts"]) == horizon * tf_ms)
    theta = cost_bps / 1e4 + k * sigma * np.sqrt(horizon)
    y = pd.Series(np.nan, index=d.index)
    y[fwd > theta] = 1.0
    y[fwd < -theta] = 0.0
    y[fwd.isna() | theta.isna()] = np.nan
    return pd.DataFrame({"fwd": fwd, "theta": theta, "y": y})

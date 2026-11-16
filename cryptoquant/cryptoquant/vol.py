"""Volatility toolkit: realized vol, HAR regression, EGARCH (fit once, filter causally)."""
import numpy as np
import pandas as pd
from arch import arch_model

WINS = (4, 96, 672, 2880)   # 1h, 1d, 1w, 30d in 15-min bars
SCALE = 100.0


def rv(r, w):
    """Per-bar realized volatility over a rolling window."""
    return np.sqrt((r ** 2).rolling(w).mean())


# ---------- HAR ----------
def _har_x(r):
    return np.log(pd.concat({w: rv(r, w) for w in WINS}, axis=1) + 1e-12)


def fit_har(r_train, horizon):
    X = _har_x(r_train)
    y = np.log(np.sqrt((r_train ** 2).rolling(horizon).mean().shift(-horizon)) + 1e-12)
    d = pd.concat([X, y.rename("y")], axis=1).dropna()
    A = np.c_[np.ones(len(d)), d[list(WINS)].values]
    return np.linalg.lstsq(A, d["y"].values, rcond=None)[0].tolist()


def har_forecast(r, b):
    X = _har_x(r)
    return pd.Series(np.exp(b[0] + X.values @ np.array(b[1:])), index=r.index)


# ---------- EGARCH ----------
def _am(r):
    return arch_model(r * SCALE, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="t", rescale=False)


def fit_garch(r_train, max_rows=30000):
    r = r_train.dropna().tail(max_rows)
    res = _am(r).fit(disp="off")
    return {k: float(v) for k, v in res.params.items()}


def garch_sigma(r, params):
    """Filter with FIXED params, so each sigma_t only uses data up to t-1: no look-ahead."""
    r = r.dropna()
    out = _am(r).fix(np.array(list(params.values())))
    return out.conditional_volatility / SCALE


# ---------- regimes ----------
def regime(har_vol, q):
    if np.isnan(har_vol):
        return "mid"
    return "low" if har_vol < q[0] else "high" if har_vol > q[1] else "mid"

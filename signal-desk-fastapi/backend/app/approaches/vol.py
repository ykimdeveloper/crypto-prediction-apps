"""Per-bar volatility forecasters. Used three ways: (1) label threshold, (2) trade gate / expected move, (3) regime label.

  rv15   rolling std of 1-bar log returns over 15 bars
  ewma   RiskMetrics: sigma^2_t = lam*sigma^2_{t-1} + (1-lam)*r_t^2           (no fitting, robust default)
  har    HAR-RV style regression on rv over 5/15/60 bars -> realized vol of the NEXT H bars (numpy OLS, fit on a training slice)
  garch  GARCH(1,1) via the optional `arch` package; params fit on the training slice, then held fixed

Returned units: per-bar log-return standard deviation (multiply by sqrt(H) for an H-bar move, by 1e4 for bps).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def rv(lr: pd.Series, n: int = 15) -> pd.Series:
    return lr.rolling(n).std()


def ewma(lr: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt((lr ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=10).mean())


def _har_X(lr: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"a": rv(lr, 5), "b": rv(lr, 15), "c": rv(lr, 60)})


def realized_forward(lr: pd.Series, horizon: int) -> pd.Series:
    """sqrt(mean(r^2)) over the NEXT `horizon` bars - the thing a vol forecast is trying to hit (NaN at the tail)."""
    return np.sqrt((lr ** 2).rolling(horizon).mean().shift(-horizon))


class HAR:
    def fit(self, lr: pd.Series, horizon: int) -> "HAR":
        X, y = _har_X(lr), realized_forward(lr, horizon)
        m = X.notna().all(axis=1) & y.notna()
        if m.sum() < 30:
            raise ValueError("not enough rows to fit HAR")
        A = np.column_stack([np.ones(m.sum()), X[m].to_numpy()])
        self.coef, *_ = np.linalg.lstsq(A, y[m].to_numpy(), rcond=None)
        return self

    def predict(self, lr: pd.Series) -> pd.Series:
        X = _har_X(lr)
        out = self.coef[0] + X.to_numpy() @ self.coef[1:]
        return pd.Series(np.maximum(out, 1e-8), index=lr.index).where(X.notna().all(axis=1))


def garch_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("arch") is not None


def garch_forecast(lr: pd.Series, horizon: int, fit_end: int) -> pd.Series:
    """Per-bar sigma over the next `horizon` bars for rows >= fit_end, with parameters estimated on rows < fit_end only.
    Needs `pip install arch`. (Not exercised by the offline unit tests - arch isn't installable in the sandbox they were written in.)"""
    from arch import arch_model
    y = (lr.fillna(0.0) * 1e4).reset_index(drop=True)                     # bps: arch behaves badly on 1e-4-sized data
    res = arch_model(y, mean="Zero", vol="GARCH", p=1, q=1).fit(last_obs=fit_end, disp="off")
    f = res.forecast(horizon=horizon, start=fit_end, reindex=True)
    sig = np.sqrt(f.variance.mean(axis=1)) / 1e4
    sig.index = lr.index
    return sig


def score_vol(frames: dict[str, pd.DataFrame], horizon: int, train_frac: float = 0.6) -> list[dict]:
    """How well does each forecaster predict the next-H-bar realized vol? Fit on the first `train_frac`, score after a purge gap."""
    methods = ("rv15", "ewma", "har", "garch")
    acc = {m: {"mae": [], "corr": [], "err": None} for m in methods}
    for sym, d in frames.items():
        lr = log_returns(d["close"])
        n = len(lr)
        split = int(n * train_frac)
        tgt = realized_forward(lr, horizon)
        idx = np.arange(n) >= split + horizon                         # forecast at t may not have been trained on t..t+H
        preds: dict[str, pd.Series | None] = {"rv15": rv(lr, 15), "ewma": ewma(lr), "har": None, "garch": None}
        try:
            preds["har"] = HAR().fit(lr.iloc[:split], horizon).predict(lr)
        except Exception as e:  # noqa: BLE001
            acc["har"]["err"] = str(e)
        if garch_available():
            try:
                preds["garch"] = garch_forecast(lr, horizon, split)
            except Exception as e:  # noqa: BLE001
                acc["garch"]["err"] = str(e)
        else:
            acc["garch"]["err"] = "arch not installed"
        for m, p in preds.items():
            if p is None:
                continue
            ok = idx & p.notna().to_numpy() & tgt.notna().to_numpy()
            if ok.sum() < 20:
                continue
            acc[m]["mae"].append(float(np.abs(p[ok] - tgt[ok]).mean() * 1e4))
            acc[m]["corr"].append(float(np.corrcoef(p[ok], tgt[ok])[0, 1]))
    return [dict(method=m, mae_bps_per_bar=(float(np.mean(a["mae"])) if a["mae"] else None),
                 corr=(float(np.mean(a["corr"])) if a["corr"] else None), note=a["err"]) for m, a in acc.items()]

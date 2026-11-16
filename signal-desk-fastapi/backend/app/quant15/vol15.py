"""Volatility for the 15m layer. Used for regimes, thresholds and the sigma comparison, NOT as a fitted input to the trees
(so the walk-forward features carry no fitted parameters)."""
from __future__ import annotations

import numpy as np
import pandas as pd

WINS = (4, 96, 672)  # 1h, 1d, 1w in 15m bars


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close.astype(float)).diff()


def ewma(lr: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt((lr ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean())


def _X(lr: pd.Series) -> pd.DataFrame:
    v = lr ** 2
    return pd.DataFrame({w: v.rolling(w).mean() for w in WINS})


class HAR15:
    """HAR in variance form: E[r_{t+1}^2] = b0 + b1*rv_1h + b2*rv_1d + b3*rv_1w. OLS on a training slice, then held fixed."""

    def __init__(self, coef=None):
        self.coef = None if coef is None else np.asarray(coef, float)

    def fit(self, lr: pd.Series) -> "HAR15":
        X, y = _X(lr), (lr ** 2).shift(-1)
        m = X.notna().all(axis=1) & y.notna()
        if m.sum() < 200:
            raise ValueError("not enough 15m bars to fit HAR (need 200+ past the one-week warm-up)")
        A = np.column_stack([np.ones(int(m.sum())), X[m].to_numpy()])
        self.coef, *_ = np.linalg.lstsq(A, y[m].to_numpy(), rcond=None)
        return self

    def predict(self, lr: pd.Series) -> pd.Series:
        """Per-bar sigma forecast for the bar AFTER each row."""
        X = _X(lr)
        out = self.coef[0] + X.to_numpy() @ self.coef[1:]
        return pd.Series(np.sqrt(np.maximum(out, 1e-12)), index=lr.index).where(X.notna().all(axis=1))


def forecast(lr: pd.Series, fit_end: int, method: str = "har") -> pd.Series:
    """Sigma forecast per row; parameters come only from rows < fit_end. garch needs `pip install arch` (falls back to har)."""
    if method == "ewma":
        return ewma(lr)
    if method == "garch":
        from ..approaches import vol as av
        if av.garch_available():
            try:
                return av.garch_forecast(lr, 1, fit_end)
            except Exception:  # noqa: BLE001
                pass
    return HAR15().fit(lr.iloc[:fit_end]).predict(lr)


def edges(sig_train: pd.Series) -> list[float]:
    return [float(x) for x in sig_train.dropna().quantile([1 / 3, 2 / 3])]


def regime(sig: float, e: list[float]) -> str:
    if sig is None or np.isnan(sig):
        return "mid"
    return "low" if sig < e[0] else "high" if sig > e[1] else "mid"

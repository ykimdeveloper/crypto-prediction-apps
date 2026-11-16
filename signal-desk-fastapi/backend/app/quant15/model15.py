"""Direction model for the NEXT 15m contract window: y = 1 if that window closes at/above its open (exactly what settles UP).

No neutral zone here on purpose: a binary contract pays on tiny moves too, so dropping them would flatter the metrics.
Models come from approaches.models (LightGBM / CatBoost / XGBoost; HistGB if none installed), validation from approaches.validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..approaches import models as am
from ..approaches.evaluate import _classification
from ..approaches.validation import walk_forward
from ..settlement import settles_up
from .bars import BAR_MS
from .features15 import FEATURES15, build_features15

PREFERRED = ["lightgbm", "catboost", "xgboost"]
MIN_BARS = 800            # one-week warm-up (672) plus something to learn from
LABEL_BARS = 2            # a row at bar t has a label that ends 2 bars later (t+1 is the window being predicted)


def pick_models(names: list[str] | None = None) -> list[str]:
    names = names or PREFERRED
    ok = [n for n in names if n in am.REGISTRY and am.REGISTRY[n].available()]
    return ok or ["hist_gb"]


def peer_symbol(symbol: str) -> str:
    base, _, quote = symbol.partition("/")
    return f"{'ETH' if base == 'BTC' else 'BTC'}/{quote or 'USDT'}"


def build_dataset15(bars: dict[str, pd.DataFrame], peers: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    peers = peers or {}
    parts = []
    for sym, b in bars.items():
        if len(b) < MIN_BARS:
            continue
        b = b.reset_index(drop=True)
        f = build_features15(b, peers.get(peer_symbol(sym)))
        nxt_ok = (b["ts"].shift(-1) - b["ts"]) == BAR_MS
        fwd = np.log(b["close"].shift(-1) / b["open"].shift(-1)).where(nxt_ok)
        y = fwd.map(lambda v: float(settles_up(v, 0.0)) if v == v else float("nan"))  # settlement.py rule: tie -> UP
        p = pd.concat([b[["ts"]], f, fwd.rename("fwd"), y.rename("y")], axis=1)
        p["symbol"] = sym
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)


def fit_ensemble(X: pd.DataFrame, y: np.ndarray, names: list[str]) -> dict:
    return {n: am.make(n).fit(X, y) for n in names}


def predict_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict_proba(X) for m in models.values()], axis=0)


def oof(data: pd.DataFrame, names: list[str], n_folds: int = 4) -> tuple[dict[str, np.ndarray], list[str]]:
    """Out-of-fold P(up) per model + 'ensemble' (mean). NaN where a row was never in a test block."""
    ts, y = data["ts"].to_numpy(), data["y"].to_numpy()
    X = data[FEATURES15]
    out = {n: np.full(len(data), np.nan) for n in names}
    skipped: list[str] = []
    for _, tr, te in walk_forward(ts, LABEL_BARS, BAR_MS, n_folds):
        trm = tr & ~np.isnan(y)
        if trm.sum() < 200 or len(np.unique(y[trm])) < 2:
            continue
        for n in names:
            try:
                out[n][te] = am.make(n).fit(X[trm], y[trm].astype(int)).predict_proba(X[te])
            except Exception as e:  # noqa: BLE001  one broken library must not sink the run
                skipped.append(f"{n}: {type(e).__name__}: {e}")
    out = {n: p for n, p in out.items() if not np.isnan(p).all()}
    if not out:
        raise ValueError("no usable walk-forward fold: need more 15m history (backfill more days)")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # rows outside every test block are all-NaN
        out["ensemble"] = np.nanmean(np.vstack(list(out.values())), axis=0)
    return out, skipped


def metrics(data: pd.DataFrame, preds: dict[str, np.ndarray]) -> list[dict]:
    y = data["y"].to_numpy()
    rows = []
    for n, p in preds.items():
        m = ~np.isnan(p) & ~np.isnan(y)
        rows.append(dict(approach=n, **_classification(y[m].astype(int), p[m])))
    r1 = data["ret_1"].to_numpy()
    m = ~np.isnan(r1) & ~np.isnan(y) & ~np.isnan(preds["ensemble"])
    if m.sum() > 30 and len(np.unique(y[m])) == 2:
        from sklearn.metrics import roc_auc_score
        rows.append(dict(approach="baseline: prev-bar direction", n=int(m.sum()), auc=float(roc_auc_score(y[m], r1[m]))))
    return sorted(rows, key=lambda r: (r.get("auc") is None, -(r.get("auc") or 0)))

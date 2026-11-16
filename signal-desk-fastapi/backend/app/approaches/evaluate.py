"""Build the shared dataset, run every approach through the same purged walk-forward folds, score out-of-fold only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import MIN_MS
from . import models, vol
from .features import FEATURES, build_features, to_frame
from .labels import make_labels
from .validation import walk_forward

MIN_LABELED = 300


def build_dataset(frames: dict[str, pd.DataFrame], horizon: int, cost_bps: float, k: float, tf_ms: int = MIN_MS) -> pd.DataFrame:
    """frames: symbol -> OHLCV frame. Returns one pooled frame: ts, symbol, FEATURES, sigma, fwd, theta, y - sorted by ts."""
    parts = []
    for sym, d in frames.items():
        if len(d) < 100:
            continue
        f = build_features(d, tf_ms)
        sigma = vol.ewma(vol.log_returns(d["close"]))
        lab = make_labels(d, sigma, horizon, cost_bps, k, tf_ms)
        p = pd.concat([d[["ts"]], f, sigma.rename("sigma"), lab], axis=1)
        p["symbol"] = sym
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["ts", "symbol"]).reset_index(drop=True)


# ---------------- trading simulation ----------------
def simulate(p: np.ndarray, fwd: np.ndarray, sigma: np.ndarray, symbols: np.ndarray, horizon: int, cost_bps: float, thr: float,
             gate_mult: float = 0.0) -> dict:
    """Long if p >= thr, short if p <= 1-thr. One position per symbol at a time (skip the next `horizon` bars), pays `cost_bps`
    round trip. gate_mult > 0: also require expected move (0.8 * sigma * sqrt(H)) >= gate_mult * cost. Rows must be time-ordered."""
    pnl: list[float] = []
    for s in np.unique(symbols):
        idx = np.flatnonzero(symbols == s)
        i = 0
        while i < len(idx):
            r = idx[i]
            side = 1 if p[r] >= thr else (-1 if p[r] <= 1 - thr else 0)
            if side and gate_mult > 0 and not (0.8 * sigma[r] * np.sqrt(horizon) * 1e4 >= gate_mult * cost_bps):
                side = 0
            if side and np.isfinite(fwd[r]):
                pnl.append(side * fwd[r] * 1e4 - cost_bps)
                i += horizon
            else:
                i += 1
    a = np.array(pnl)
    if a.size == 0:
        return dict(n_trades=0, net_bps_total=0.0, avg_net_bps=None, hit_rate=None, profit_factor=None, max_drawdown_bps=0.0)
    cum = np.cumsum(a)
    wins, losses = a[a > 0].sum(), -a[a < 0].sum()
    return dict(n_trades=int(a.size), net_bps_total=float(a.sum()), avg_net_bps=float(a.mean()), hit_rate=float((a > 0).mean()),
                profit_factor=(float(wins / losses) if losses > 0 else None),
                max_drawdown_bps=float((np.maximum.accumulate(cum) - cum).max()))


def _classification(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    if len(y) < 30 or len(np.unique(y)) < 2:
        return dict(n=int(len(y)), auc=None, brier=None, logloss=None, accuracy=None)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(n=int(len(y)), auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, pc)),
                logloss=float(log_loss(y, pc)), accuracy=float(((p >= 0.5) == (y == 1)).mean()))


def compare(data: pd.DataFrame, horizon: int, cost_bps: float, approaches: list[str] | None = None, n_folds: int = 4,
            thr: float = 0.55, gate_mult: float = 1.5, tf_ms: int = MIN_MS, progress=None) -> dict:
    approaches = approaches or models.DEFAULT_COMPARE
    n_lab = int(data["y"].notna().sum()) if len(data) else 0
    if n_lab < MIN_LABELED:
        raise ValueError(f"only {n_lab} labeled rows (need >= {MIN_LABELED}). Let the feed run longer, lower cost_bps / k, or add symbols.")
    ts = data["ts"].to_numpy()
    y_all = data["y"].to_numpy()
    X = data[FEATURES]
    oof = {a: np.full(len(data), np.nan) for a in approaches}
    fold_auc: dict[str, list] = {a: [] for a in approaches}
    skipped: dict[str, str] = {a: "" for a in approaches}
    folds_used = 0
    for j, tr, te in walk_forward(ts, horizon, tf_ms, n_folds):
        trm = tr & ~np.isnan(y_all)
        if trm.sum() < 100 or len(np.unique(y_all[trm])) < 2:
            continue
        folds_used += 1
        for a in approaches:
            cls = models.REGISTRY[a]
            if not cls.available():
                skipped[a] = f"{cls.needs} not installed"
                continue
            if progress:
                progress(f"fold {j + 1}/{n_folds} · {a}")
            try:
                m = models.make(a).fit(X[trm], y_all[trm].astype(int))
                oof[a][te] = m.predict_proba(X[te])
            except Exception as e:  # noqa: BLE001  one broken library must not sink the whole comparison
                skipped[a] = f"{type(e).__name__}: {e}"
                continue
            lab = te & ~np.isnan(y_all)
            r = _classification(y_all[lab].astype(int), oof[a][lab])
            fold_auc[a].append(r["auc"])
    if not folds_used:
        raise ValueError("no usable walk-forward fold (training windows too small or one-sided). Need more history.")

    learned = [a for a in approaches if models.REGISTRY[a].fitted and not np.isnan(oof[a]).all()]
    if len(learned) >= 2:
        st = np.vstack([oof[a] for a in learned])
        oof["ensemble_avg"] = np.where(np.isnan(st).all(axis=0), np.nan, np.nanmean(np.where(np.isnan(st), 0, st), axis=0) * st.shape[0] / np.maximum((~np.isnan(st)).sum(axis=0), 1))
        fold_auc["ensemble_avg"], skipped["ensemble_avg"] = [], ""

    fwd, sigma, sym = data["fwd"].to_numpy(), data["sigma"].fillna(0).to_numpy(), data["symbol"].to_numpy()
    results = []
    for a, p in oof.items():
        cls = models.REGISTRY.get(a)
        have = ~np.isnan(p)
        if not have.any():
            results.append(dict(approach=a, label=cls.label if cls else a, available=False, note=skipped.get(a), auc=None))
            continue
        lab = have & ~np.isnan(y_all)
        cl = _classification(y_all[lab].astype(int), p[lab])
        sel = have & np.isfinite(fwd)
        pp, ff, ss, yy = p[sel], fwd[sel], sigma[sel], sym[sel]
        results.append(dict(approach=a, label=cls.label if cls else "Average of learned models", kind=cls.kind if cls else "ensemble",
                            available=True, note=skipped.get(a) or None, **cl, fold_auc=fold_auc[a],
                            trading=simulate(pp, ff, ss, yy, horizon, cost_bps, thr, 0.0),
                            trading_gated=simulate(pp, ff, ss, yy, horizon, cost_bps, thr, gate_mult)))
    results.sort(key=lambda r: (r["auc"] is None, -(r["auc"] or 0)))

    warnings = []
    if n_lab / horizon < 300:
        warnings.append(f"~{int(n_lab / horizon)} effectively independent labels (labeled rows / horizon). Metrics are very noisy; treat rankings as anecdotal.")
    best = next((r for r in results if r["auc"] is not None and r.get("kind") != "baseline"), None)
    if best and best["auc"] < 0.52:
        warnings.append("Best learned AUC < 0.52: no detectable edge over a coin flip after purging. Do not trade this.")
    if len(np.unique(sym)) == 1:
        warnings.append("Single symbol: results may not generalise.")
    return dict(rows=int(len(data)), labeled=n_lab, up_share=float(np.nanmean(y_all)), folds=folds_used, results=results, warnings=warnings)

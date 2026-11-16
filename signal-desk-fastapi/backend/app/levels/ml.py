"""ML layer: does price REACT (hold) when it next touches a zone, and how far should each (timeframe, span) cell be trusted.

Unit of training = one touch event. Label = held (1) vs broke (0); inconclusive and pending are excluded.
How events are built without look-ahead: at each checkpoint `cp`, zones are detected from candles closed before cp (the same code the live
API uses). Only touches in [cp, cp + step) become events, so every event is out-of-sample relative to its zone's definition and each
moment belongs to exactly one checkpoint. Features use only what was known at the touch time.

Validation: expanding-window walk-forward purged on each row's OUTCOME END (touches of one zone inside one horizon are correlated), a
final holdout (the newest `holdout_frac` of time) that no model, calibrator or threshold ever sees, isotonic calibration on the newest
slice of the development data, and a cluster bootstrap so intervals use the effective sample size, not the raw row count.
Baselines: random levels (same width, same number, same checkpoints) and the previous day's high/low. If the ML zones do not beat them
the report says so. Every zone carries reaction_prob, n_touches, an interval and edge = measured | unproven | noise.

Honest limits: 75 cells produce false winners, so intervals are shown, cells are ranked by an interval lower bound, and a cell whose
interval includes "no edge" is noise. The deployed model is trained on development data only (not on the holdout), on purpose."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import confluence as CF, features as F, service, touches as T
from .data import SPAN_MS, TF_MS
from .candidates import atr

W = 900_000
MIN_EFF = F.MIN_EFF
MIN_DEV_EVENTS, MIN_HOLDOUT_EVENTS = 200, 50
PREVDAY = {"prev_day_high", "prev_day_low"}
RELIABILITY_SCALE = 0.10     # heuristic: cell weight = clip((lower bound of edge vs random + 0.10) / 0.20, 0, 1)
N_BOOT = 400


# ------------------------------------------------------------------ events
def build_events(frames, symbol, cfg, tfs, spans, n_cp=40, step_ms=6 * 3_600_000, seed=1):
    """frames: {tf: OHLCV frame}. Returns one row per resolved touch (real zones, random zones, previous-day zones)."""
    rng = np.random.default_rng(seed)
    cfg0 = {**cfg, "min_touches": 0}
    out_ms = lambda tf: int((T.horizon_bars(TF_MS[tf], int(cfg["horizon_min"] * 60_000)) + 1) * TF_MS[tf])
    t_end = min(int(f["ts"].iloc[-1]) + TF_MS[tf] for tf, f in frames.items() if len(f))
    cps = [(t_end - (n_cp - j) * step_ms) // W * W for j in range(n_cp)]
    rows = []
    for cp in cps:
        cells = [service.compute_cell(frames[tf], tf, sp, cp, cfg0) for tf in tfs for sp in spans]
        conf = {i: g["confluence_count"] for g in CF.merge(cells) for i in g["zone_ids"]}
        for c in cells:
            if c["status"] != "ok":
                continue
            tf, sp = c["tf"], c["span"]
            tf_ms, span_ms = TF_MS[tf], SPAN_MS[sp]
            f = frames[tf]
            ext = f[(f["ts"] + tf_ms <= cp + step_ms) & (f["ts"] >= cp - span_ms)].reset_index(drop=True)
            a = atr(ext)
            ts, cl = ext["ts"].to_numpy(), ext["close"].to_numpy()
            rv, reg = F.vol_series(ext["close"])
            rv, reg, av = rv.to_numpy(), reg.to_numpy(), a.to_numpy()
            lo_all, hi_all = float(ext["low"][ext["ts"] < cp].min()), float(ext["high"][ext["ts"] < cp].max())
            width = cfg["zone_width_atr"] * c["atr"]
            rand = [dict(center=x, low=x - width / 2, high=x + width / 2, scan_from=cp - 1, sources=["random"], width_atr=width / c["atr"], vol_share=0.0)
                    for x in rng.uniform(lo_all, hi_all, size=max(1, len(c["zones"])))]
            for kind, zs in (("real", c["zones"]), ("random", rand)):
                for z in zs:
                    tl, _ = T.touch_log(ext, z, a, tf, tf_ms, int(cfg["horizon_min"] * 60_000), cfg["x_atr"])
                    zid = z.get("id") or f"rand:{tf}:{sp}:{z['center']:.2f}"
                    for t in tl:
                        if not (cp <= t["ts"] < cp + step_ms) or t["outcome"] not in ("held", "broke"):
                            continue
                        i = int(np.searchsorted(ts, t["ts"]))
                        below = t["approach"] == "from_below"
                        edge_gap = (z["low"] - cl[i - 1]) if below else (cl[i - 1] - z["high"])
                        row = dict(symbol=symbol, cell=f"{tf}/{sp}", tf=tf, span=sp, zone_id=zid, cp=cp, ts=t["ts"], end=t["ts"] + out_ms(tf), y=float(t["outcome"] == "held"),
                                   is_random=float(kind == "random"), is_prevday=float(kind == "real" and set(z["sources"]) <= PREVDAY))
                        if kind == "real":
                            prior = F.prior_stats(tl, t["ts"], out_ms(tf))
                            ctx = dict(rv_20=None if rv[i - 1] != rv[i - 1] else float(rv[i - 1]), vol_regime=None if reg[i - 1] != reg[i - 1] else float(reg[i - 1]))
                            row.update(F.feature_row(z, ctx, t["approach"], t["ts"], prior, conf.get(zid, 1), float(edge_gap / av[i]), tf, span_ms))
                        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ validation helpers
def purged_folds(ts, end, n_folds=4, min_train_frac=0.5):
    """Expanding window on timestamps; a training row is used only if its outcome ended before the test block starts."""
    ts, end = np.asarray(ts, dtype="int64"), np.asarray(end, dtype="int64")
    u = np.unique(ts)
    if len(u) < n_folds * 4:
        raise ValueError(f"only {len(u)} distinct timestamps for {n_folds} folds")
    start = int(len(u) * min_train_frac)
    block = max(1, (len(u) - start) // n_folds)
    for j in range(n_folds):
        a = u[start + j * block]
        b = u[start + (j + 1) * block] if j < n_folds - 1 else u[-1] + 1
        yield j, end < a, (ts >= a) & (ts < b)


def clusters(df):
    """Cluster id per row: touches of the same zone (same checkpoint) inside one outcome window are one independent observation."""
    if df.empty:
        return np.array([], dtype=int)
    o = df.sort_values(["symbol", "zone_id", "cp", "ts"])
    key = o["symbol"].astype(str) + "|" + o["zone_id"].astype(str) + "|" + o["cp"].astype(str)
    new = (key != key.shift()).to_numpy()
    ts, end = o["ts"].to_numpy(), o["end"].to_numpy()
    cid, start_ts, start_end = -1, 0, 0
    ids = np.empty(len(o), dtype=int)
    for k in range(len(o)):
        if new[k] or ts[k] >= start_end:
            cid += 1
            start_ts, start_end = ts[k], end[k]
        ids[k] = cid
    return pd.Series(ids, index=o.index).reindex(df.index).to_numpy()


def _agg(y, cid):
    """Per-cluster sums and counts. Cluster ids are re-labelled first: a subset of rows must not carry empty clusters into the bootstrap."""
    _, inv = np.unique(np.asarray(cid), return_inverse=True)
    return np.bincount(inv, weights=y), np.bincount(inv).astype(float)


def boot_rate(y, cid, n_boot=N_BOOT, seed=7):
    y, cid = np.asarray(y, float), np.asarray(cid)
    if len(y) == 0:
        return dict(rate=None, lo=None, hi=None, n=0, n_eff=0)
    s, c = _agg(y, cid)
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(s), size=(n_boot, len(s)))
    b = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return dict(rate=float(y.mean()), lo=float(np.quantile(b, 0.05)), hi=float(np.quantile(b, 0.95)), n=int(len(y)), n_eff=int(len(s)))


def boot_diff(y1, c1, y0, c0, n_boot=N_BOOT, seed=7):
    """90% interval for rate(y1) - rate(y0), resampling clusters independently in each group."""
    if len(y1) == 0 or len(y0) == 0:
        return None
    s1, n1 = _agg(np.asarray(y1, float), np.asarray(c1))
    s0, n0 = _agg(np.asarray(y0, float), np.asarray(c0))
    r = np.random.default_rng(seed)
    i1 = r.integers(0, len(s1), size=(n_boot, len(s1)))
    i0 = r.integers(0, len(s0), size=(n_boot, len(s0)))
    d = s1[i1].sum(1) / np.maximum(n1[i1].sum(1), 1) - s0[i0].sum(1) / np.maximum(n0[i0].sum(1), 1)
    return dict(diff=float(np.mean(y1) - np.mean(y0)), lo=float(np.quantile(d, 0.05)), hi=float(np.quantile(d, 0.95)))


def edge_flag(n_eff, diff):
    if n_eff < MIN_EFF or diff is None:
        return "unproven"
    return "measured" if diff["lo"] > 0 else "noise"


def reliability(diff):
    return 0.0 if diff is None else float(np.clip((diff["lo"] + RELIABILITY_SCALE) / (2 * RELIABILITY_SCALE), 0, 1))


def cell_table(real, rand, pred=None):
    """Per-cell reaction rate + interval, effective sample size, edge vs random levels, calibration gap, edge flag."""
    out = {}
    rc, rnc = clusters(real), clusters(rand)
    for cell in sorted(real["cell"].unique()):
        m = (real["cell"] == cell).to_numpy()
        mr = (rand["cell"] == cell).to_numpy() if len(rand) else np.array([], bool)
        base = rand[mr] if mr.sum() else rand
        bc = clusters(base) if len(base) else np.array([], int)
        rate = boot_rate(real["y"][m], rc[m])
        diff = boot_diff(real["y"][m], rc[m], base["y"], bc) if len(base) else None
        out[cell] = dict(**rate, random_rate=float(base["y"].mean()) if len(base) else None, edge_vs_random=diff, edge=edge_flag(rate["n_eff"], diff),
                         reliability=reliability(diff), calibration_gap=(float(np.mean(pred[m]) - real["y"][m].mean()) if pred is not None and m.any() else None))
    return out


# ------------------------------------------------------------------ training
def _model(name):
    from ..approaches import models as M
    n = name if name and M.REGISTRY[name].available() else ("lightgbm" if M.LightGBM.available() else "hist_gb")
    return n, M.make(n)


def fit_and_validate(ev, holdout_frac=0.2, n_folds=4, approach=None):
    real, rand = ev[ev["is_random"] == 0].reset_index(drop=True), ev[ev["is_random"] == 1].reset_index(drop=True)
    cut = int(np.quantile(np.unique(real["ts"]), 1 - holdout_frac))
    dev = real[(real["ts"] < cut) & (real["end"] < cut)].reset_index(drop=True)      # purge: no dev outcome reaches into the holdout
    hold = real[real["ts"] >= cut].reset_index(drop=True)
    info = dict(events=int(len(real)), dev_events=int(len(dev)), holdout_events=int(len(hold)), holdout_start=cut, random_events=int(len(rand)))
    if len(dev) < MIN_DEV_EVENTS or len(hold) < MIN_HOLDOUT_EVENTS or dev["y"].nunique() < 2:
        return None, dict(status="insufficient_events", need=dict(dev=MIN_DEV_EVENTS, holdout=MIN_HOLDOUT_EVENTS), **info)
    name, _ = _model(approach)
    X = lambda d: d[F.FEATURES]
    # 1) out-of-fold predictions on dev (weights + per-cell calibration), purged on outcome end
    oof = np.full(len(dev), np.nan)
    for _, tr, te in purged_folds(dev["ts"], dev["end"], n_folds):
        if tr.sum() >= 100 and dev["y"][tr].nunique() == 2 and te.any():
            oof[te] = _model(name)[1].fit(X(dev[tr]), dev["y"][tr].to_numpy()).predict_proba(X(dev[te]))
    have = ~np.isnan(oof)
    # 2) final model on the older part of dev, isotonic on the newest slice of dev; holdout untouched
    cal_cut = int(np.quantile(dev["ts"], 0.75))
    fit_m, cal_m = ((dev["ts"] < cal_cut) & (dev["end"] < cal_cut)).to_numpy(), (dev["ts"] >= cal_cut).to_numpy()
    model = _model(name)[1].fit(X(dev[fit_m]), dev["y"][fit_m].to_numpy())
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(model.predict_proba(X(dev[cal_m])), dev["y"][cal_m].to_numpy())
    p_cal = iso.predict(model.predict_proba(X(dev[cal_m])))
    p_hold = iso.predict(model.predict_proba(X(hold)))
    thr = float(np.quantile(p_cal, 2 / 3))        # 'top third' threshold comes from dev data, never from the holdout
    bundle = dict(model=model, iso=iso, features=F.FEATURES, approach=name, top_third_threshold=thr)
    return (bundle, dev, hold, rand, oof, have, p_hold, info), None


def _prob_metrics(y, p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    base = np.full(len(y), float(np.mean(y)))
    q = np.clip(p, 1e-4, 1 - 1e-4)
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), base_rate=float(np.mean(y)), logloss=float(log_loss(y, q, labels=[0, 1])) if both else None,
                logloss_base_rate=float(log_loss(y, np.clip(base, 1e-4, 1 - 1e-4), labels=[0, 1])) if both else None,
                brier=float(brier_score_loss(y, p)), brier_base_rate=float(brier_score_loss(y, base)), auc=float(roc_auc_score(y, p)) if both else None)


def train_report(ev, holdout_frac=0.2, n_folds=4, approach=None):
    res, bad = fit_and_validate(ev, holdout_frac, n_folds, approach)
    if bad:
        return None, bad
    bundle, dev, hold, rand, oof, have, p_hold, info = res
    t_lo = int(hold["ts"].min())
    rand_hold, rand_dev = rand[rand["ts"] >= t_lo].reset_index(drop=True), rand[rand["ts"] < t_lo].reset_index(drop=True)
    oof_rows, oof_p = dev[have].reset_index(drop=True), oof[have]
    cells_oof = cell_table(oof_rows, rand_dev[rand_dev["ts"] >= (oof_rows["ts"].min() if len(oof_rows) else 0)], oof_p)
    cells_hold = cell_table(hold, rand_hold, p_hold)
    # headline (holdout only)
    hc = clusters(hold)
    top = (p_hold >= bundle["top_third_threshold"])
    rc = clusters(rand_hold) if len(rand_hold) else np.array([], int)
    pd_all = ev[(ev["is_prevday"] == 1) & (ev["ts"] >= t_lo)].reset_index(drop=True)
    pc = clusters(pd_all) if len(pd_all) else np.array([], int)
    vs_random = boot_diff(hold["y"][top], hc[top], rand_hold["y"], rc) if top.any() and len(rand_hold) else None
    vs_prev = boot_diff(hold["y"][top], hc[top], pd_all["y"], pc) if top.any() and len(pd_all) else None
    def verdict(name, d):
        if d is None:
            return f"ML top-third vs {name}: not enough events to compare."
        beat = d["lo"] > 0
        return (f"ML top-third zones {'BEAT' if beat else 'do NOT beat'} {name}: held-rate difference {d['diff']:+.3f} (90% interval {d['lo']:+.3f} to {d['hi']:+.3f})"
                + ("." if beat else "; the interval includes zero, so no edge is shown."))
    metrics = _prob_metrics(hold["y"].to_numpy(), p_hold)
    verdicts = [verdict("random levels", vs_random), verdict("previous-day high/low", vs_prev)]
    if metrics["logloss"] is not None:
        verdicts.append(f"Model log-loss on the holdout is {'better' if metrics['logloss'] < metrics['logloss_base_rate'] else 'NOT better'} than always predicting the base rate "
                        f"({metrics['logloss']:.4f} vs {metrics['logloss_base_rate']:.4f}).")
    key = lambda kv: -((kv[1]["edge_vs_random"] or {}).get("lo", -9))
    report = dict(status="ok", trained_ms=int(time.time() * 1000), approach=bundle["approach"], **info, headline_source="holdout (never seen by the model, calibrator or thresholds)",
                  holdout=dict(metrics=metrics, top_third_threshold=bundle["top_third_threshold"], top_third_events=int(top.sum()),
                               top_third_vs_random=vs_random, top_third_vs_prev_day=vs_prev, prev_day_events=int(len(pd_all)), prev_day_rate=float(pd_all["y"].mean()) if len(pd_all) else None,
                               random_rate=float(rand_hold["y"].mean()) if len(rand_hold) else None),
                  verdicts=verdicts, cells_holdout=dict(sorted(cells_hold.items(), key=key)), cells_oof=dict(sorted(cells_oof.items(), key=key)),
                  ranking="cells ranked by the lower bound of (held rate - random-level held rate), 90% cluster-bootstrap interval",
                  warnings=[f"{len(cells_oof) or len(cells_hold)} cells were tested; some will look good by chance, so trust the interval, not the point estimate.",
                            f"A cell with fewer than {MIN_EFF} independent observations (n_eff) is 'unproven'.", "n_eff counts independent clusters, not raw touches."])
    bundle["cells"] = {c: dict(**cells_oof.get(c, {}), holdout=cells_hold.get(c)) for c in set(cells_oof) | set(cells_hold)}
    return bundle, report

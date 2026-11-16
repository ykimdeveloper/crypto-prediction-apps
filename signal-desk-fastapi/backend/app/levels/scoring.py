"""Attach the trained model's output to live zones. Read-only; never changes which zones exist. A zone carries reaction_prob (calibrated
P(held) on its next touch), n_touches, cell-level reaction-rate interval, effective sample size and an edge flag. Without a trained model
for the exchange every zone stays edge='unproven'."""
from __future__ import annotations

import json
import re
import threading

import numpy as np
import pandas as pd

from .. import config as app_config
from . import features as F
from .features import MIN_EFF

_cache: dict = {}
_lock = threading.Lock()


def _safe(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)[:60]


def bundle_path(exchange):
    return app_config.MODEL_DIR / f"levels_model_{_safe(exchange)}.joblib"


def report_path(exchange):
    return app_config.MODEL_DIR / f"levels_report_{_safe(exchange)}.json"


def load_bundle(exchange):
    p = bundle_path(exchange)
    if not p.exists():
        return None
    m = p.stat().st_mtime
    with _lock:
        hit = _cache.get(exchange)
        if hit and hit[0] == m:
            return hit[1]
    from joblib import load
    b = load(p)
    with _lock:
        _cache[exchange] = (m, b)
    return b


def forget(exchange):
    with _lock:
        _cache.pop(exchange, None)


def load_report(exchange):
    p = report_path(exchange)
    return json.loads(p.read_text()) if p.exists() else None


def cell_view(bundle, cell):
    """(edge, interval block, n_eff, source) for a cell: the holdout when it has enough independent observations, else the dev-only view."""
    c = (bundle.get("cells") or {}).get(cell)
    if not c:
        return "unproven", None, 0, "cell not in the trained grid"
    h = c.get("holdout")
    if h and h["n_eff"] >= MIN_EFF:
        return h["edge"], h, h["n_eff"], "holdout"
    return "unproven", c, c["n_eff"], f"holdout has only {h['n_eff'] if h else 0} independent observations (dev-only flag: {c['edge']})"


def weights(bundle):
    return {c: v["reliability"] for c, v in (bundle.get("cells") or {}).items()} if bundle else {}


def attach(cells, conf_counts, bundle, as_of):
    rows, refs = [], []
    for c in cells:
        if c["status"] != "ok":
            continue
        for z in c["zones"]:
            below_price = z["side"] == "above" or (z["side"] == "inside" and c["close"] <= z["center"])
            gap = (z["low"] - c["close"]) if z["side"] == "above" else (c["close"] - z["high"]) if z["side"] == "below" else 0.0
            done = z["held"] + z["broke"]
            prior = dict(n=z["n_touches"], held_rate=(z["held"] / done) if done else None, last_ts=z["last_touch"])
            from .data import SPAN_MS
            rows.append(F.feature_row(z, dict(rv_20=c.get("rv_20"), vol_regime=c.get("vol_regime")), "from_below" if below_price else "from_above", as_of, prior,
                                      conf_counts.get(z["id"], 1), float(gap / c["atr"]), c["tf"], SPAN_MS[c["span"]]))
            refs.append(z)
    if not rows:
        return
    p = bundle["iso"].predict(bundle["model"].predict_proba(pd.DataFrame(rows)[bundle["features"]]))
    for z, pr in zip(refs, p):
        edge, blk, n_eff, src = cell_view(bundle, z["cell"])
        z.update(reaction_prob=float(pr), edge=edge, cell_reaction_rate=(dict(rate=blk["rate"], lo=blk["lo"], hi=blk["hi"]) if blk else None),
                 cell_n_eff=int(n_eff), edge_note=f"cell {z['cell']}: {src}",
                 prob_note="reaction_prob = calibrated P(held) at the next touch; the interval is the cell's measured reaction rate, not an interval on this probability")

"""One (timeframe, span) cell -> zones with their evidence; several cells -> the /api/levels response.
Not here yet (later build steps): ML reaction_prob/edge flags, HAR expected move.
Until the model exists every zone is labeled edge='unproven'."""
from __future__ import annotations

import copy
import threading
import time

import numpy as np

from . import scoring, candidates as C, confluence as CF, features as F, touches as T, windows as WN
from .data import SPAN_MS, TF_MS

MIN_CANDLES = 40
MAX_MISSING_FRAC = 0.10
MIN_COVERED_FRAC = 0.80
HEAVY_CANDLES = 200_000
MAX_TOUCHES_IN_LIST = 25
_cache: dict = {}
_lock = threading.Lock()


def zone_id(tf, span, center):
    return f"{tf}:{span}:{center:.2f}"


def gate(d_all, tf, span, as_of):
    """Closed candles inside the span + the honest status of the cell: (d, cell, None) when usable, else (d, cell, cell_with_reason)."""
    tf_ms, span_ms = TF_MS[tf], SPAN_MS[span]
    d = d_all[(d_all["ts"] + tf_ms <= as_of) & (d_all["ts"] >= as_of - span_ms)].reset_index(drop=True)   # closed candles only
    cell = dict(tf=tf, span=span, as_of=int(as_of), n_candles=len(d), heavy=len(d) > HEAVY_CANDLES, status="ok", zones=[])
    if len(d) < MIN_CANDLES:
        return d, cell, {**cell, "status": "not_enough_structure", "note": f"{len(d)} candles; need at least {MIN_CANDLES}"}
    first, last = int(d["ts"].iloc[0]), int(d["ts"].iloc[-1])
    expected = (last - first) // tf_ms + 1
    cov = dict(first=first, last=last, expected=int(expected), missing=int(expected - len(d)),
               missing_frac=float((expected - len(d)) / expected), covered_frac=float(min(1.0, (last + tf_ms - first) / span_ms)))
    cell["coverage"] = cov
    if cov["covered_frac"] < MIN_COVERED_FRAC:
        return d, cell, {**cell, "status": "short_history", "note": f"stored candles cover {cov['covered_frac']:.0%} of the span"}
    if cov["missing_frac"] > MAX_MISSING_FRAC:
        return d, cell, {**cell, "status": "gapped", "note": f"{cov['missing']} of {expected} candles are missing ({cov['missing_frac']:.0%})"}
    return d, cell, None


def compute_cell(d_all, tf, span, as_of, cfg, all_touches=False, bars15=None, n_windows=None):
    tf_ms = TF_MS[tf]
    d, cell, bad = gate(d_all, tf, span, as_of)
    if bad:
        return bad
    first = int(d["ts"].iloc[0])
    a = C.atr(d)
    a_last = float(a.iloc[-1])
    if not a_last > 0:
        return {**cell, "status": "not_enough_structure", "note": "ATR is not defined yet"}
    cands = C.swing_pivots(d, a, tf_ms) + C.prior_period(d, as_of) + C.volume_nodes(d, a_last)
    if cfg.get("round_numbers"):
        cands += C.round_numbers(d, first)
    close = float(d["close"].iloc[-1])
    grid = WN.window_grid(bars15, as_of, n_windows) if n_windows and bars15 is not None and len(bars15) else None
    zones = []
    for z in C.cluster(cands, cfg["zone_width_atr"] * a_last):
        tl, flips = T.touch_log(d, z, a, tf, tf_ms, int(cfg["horizon_min"] * 60_000), cfg["x_atr"])
        stats = T.summarize(tl, flips, z["scan_from"])
        if stats["n_touches"] < cfg["min_touches"]:
            continue
        ws = WN.zone_stats(z, *grid, tl, all_touches) if grid else dict(status="no_15m_candles", n_windows=n_windows)
        mid = (z["low"] + z["high"]) / 2
        dist = mid - close
        ov = (d["low"] <= z["high"]) & (d["high"] >= z["low"])
        vshare = float(d["volume"][ov].sum() / d["volume"].sum()) if d["volume"].sum() > 0 else 0.0
        zones.append(dict(id=zone_id(tf, span, z["center"]), cell=f"{tf}/{span}", **z, **stats, window_stats=ws, width_atr=float((z["high"] - z["low"]) / a_last), vol_share=vshare, edge="unproven",
                          edge_note="no model has scored this zone yet", side="above" if z["low"] > close else "below" if z["high"] < close else "inside",
                          distance=dict(price=float(dist), bp=float(dist / close * 1e4), atr=float(dist / a_last)),
                          touches=tl if all_touches else tl[-MAX_TOUCHES_IN_LIST:], role_flips=flips,
                          touches_truncated=(not all_touches and len(tl) > MAX_TOUCHES_IN_LIST)))
    above = [z for z in zones if z["side"] == "above"]
    below = [z for z in zones if z["side"] == "below"]
    rv, reg = F.vol_series(d["close"])
    return {**cell, "close": close, "atr": a_last, "rv_20": F.last(rv), "vol_regime": F.last(reg), "zones": zones,
            "nearest_above": min(above, key=lambda z: z["low"] - close)["id"] if above else None,
            "nearest_below": max(below, key=lambda z: z["high"])["id"] if below else None}


def _cell(exchange, symbol, tf, span, as_of, cfg, loader, all_touches=False, bars15=None, n_windows=None):
    key = (exchange, symbol, tf, span, as_of // 60_000, cfg["min_touches"], cfg["horizon_min"], cfg["x_atr"], cfg["zone_width_atr"],
           cfg["round_numbers"], all_touches, n_windows)
    with _lock:
        hit = _cache.get(key)
    if hit:
        return hit
    res = compute_cell(loader(exchange, symbol, tf), tf, span, as_of, cfg, all_touches, bars15, n_windows)
    with _lock:
        if len(_cache) > 512:
            _cache.clear()
        _cache[key] = res
    return res


def levels_for(exchange, symbol, cfg, tfs=None, spans=None, as_of=None, loader=None, windows=None):
    if loader is None:
        from .data import load as loader
    as_of = int(as_of if as_of is not None else time.time() * 1000)
    tfs, spans = tfs or cfg["tfs"], spans or cfg["spans"]
    n = int(windows or cfg["windows"])
    bars15 = loader(exchange, symbol, "15m")
    cells = [_cell(exchange, symbol, tf, sp, as_of, cfg, loader, False, bars15, n) for tf in tfs for sp in spans]
    conf = CF.merge(cells)
    bundle = scoring.load_bundle(exchange)
    if bundle is not None:
        cells = copy.deepcopy(cells)     # cached cells stay pristine
        scoring.attach(cells, {i: g["confluence_count"] for g in conf for i in g["zone_ids"]}, bundle, as_of)
        conf = CF.merge(cells, scoring.weights(bundle))
    if cfg.get("confluence_only"):
        conf = [g for g in conf if g["confluence_count"] >= 2]
        keep = {i for g in conf for i in g["zone_ids"]}
        cells = [{**c, "zones": [z for z in c["zones"] if z["id"] in keep]} if c["status"] == "ok" else c for c in cells]
    return dict(enabled=True, informational_only=True, exchange=exchange, symbol=symbol, as_of=as_of, windows=n, cells=cells, confluence=conf,
                model=dict(loaded=bundle is not None, approach=bundle["approach"] if bundle else None),
                pending_features=["HAR expected move / distance to strike"])


def zone_detail(exchange, symbol, zid, cfg, as_of=None, loader=None, windows=None):
    if loader is None:
        from .data import load as loader
    tf, span, _ = zid.split(":")
    as_of = int(as_of if as_of is not None else time.time() * 1000)
    c = _cell(exchange, symbol, tf, span, as_of, {**cfg, "min_touches": 0}, loader, True, loader(exchange, symbol, "15m"), int(windows or cfg["windows"]))
    hits = [z for z in c["zones"] if z["id"] == zid]
    return dict(cell={k: v for k, v in c.items() if k != "zones"}, zone=hits[0] if hits else None)


def grid(exchange, symbol, cfg, as_of=None, loader=None, zones=False):
    """The timeframe x span heatmap (all 5 x 15 cells): honest status per cell + the trained reliability and edge flag.
    Zones are only counted when zones=True (long spans on 1m candles are heavy, so they are computed on demand)."""
    if loader is None:
        from .data import load as loader
    as_of = int(as_of if as_of is not None else time.time() * 1000)
    bundle = scoring.load_bundle(exchange)
    out = []
    for tf in TF_MS:
        d_all = loader(exchange, symbol, tf)
        for sp in SPAN_MS:
            if sp == "1y":
                continue        # alias of 12mo
            d, cell, bad = gate(d_all, tf, sp, as_of)
            row = dict(tf=tf, span=sp, status=(bad or cell)["status"], n_candles=cell["n_candles"], heavy=cell["heavy"], coverage=cell.get("coverage"),
                       note=(bad or {}).get("note"), n_zones=None, reliability=None, edge="unproven", edge_source=None)
            if bundle is not None:
                edge, blk, n_eff, src = scoring.cell_view(bundle, f"{tf}/{sp}")
                row.update(edge=edge, edge_source=src, reliability=(bundle.get("cells") or {}).get(f"{tf}/{sp}", {}).get("reliability"), n_eff=n_eff,
                           cell_reaction_rate=(dict(rate=blk["rate"], lo=blk["lo"], hi=blk["hi"]) if blk else None))
            if zones and row["status"] == "ok":
                row["n_zones"] = len(compute_cell(d_all, tf, sp, as_of, {**cfg, "min_touches": 0})["zones"])
            out.append(row)
    return dict(exchange=exchange, symbol=symbol, as_of=as_of, cells=out, model_loaded=bundle is not None,
                colors="by edge flag (measured / unproven / noise); thin, gapped and short-history cells are grey")

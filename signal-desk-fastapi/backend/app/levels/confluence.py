"""Merge zones across cells when their edges overlap. Each cell counts once per merged zone. Until the ML step provides
out-of-fold reliability per cell, every cell has weight 1.0 (stated in `weights`), capped so no single cell can dominate."""
from __future__ import annotations

WEIGHT_CAP = 1.0


def merge(cells, weights=None):
    weights = weights or {}
    zs = [z for c in cells if c["status"] == "ok" for z in c["zones"]]
    zs.sort(key=lambda z: z["low"])
    groups, cur, hi = [], [], None
    for z in zs:
        if cur and z["low"] <= hi:
            cur.append(z)
            hi = max(hi, z["high"])
        else:
            if cur:
                groups.append(cur)
            cur, hi = [z], z["high"]
    if cur:
        groups.append(cur)
    close = next((c["close"] for c in cells if c["status"] == "ok"), None)
    out = []
    for g in groups:
        by_cell = {}
        for z in g:
            by_cell.setdefault(z["cell"], z)
        tight = min(g, key=lambda z: z["high"] - z["low"])
        low, high = min(z["low"] for z in g), max(z["high"] for z in g)
        m = dict(id="conf:" + ":".join(sorted(z["id"] for z in by_cell.values()))[:120], low=low, high=high, center=(low + high) / 2,
                 tightest=dict(low=tight["low"], high=tight["high"]), cells=sorted(by_cell), confluence_count=len(by_cell),
                 score=float(sum(min(weights.get(c, 1.0), WEIGHT_CAP) for c in by_cell)),
                 weights="equal per cell (no reliability model yet)" if not weights else "out-of-fold cell reliability, capped",
                 zone_ids=[z["id"] for z in g], n_touches_by_cell={c: z["n_touches"] for c, z in by_cell.items()},
                 touches_note="the same candles can appear in several cells; touch counts are not additive")
        if close:
            mid = m["center"]
            m["side"] = "above" if low > close else "below" if high < close else "inside"
            m["distance"] = dict(price=mid - close, bp=(mid - close) / close * 1e4)
        out.append(m)
    return out

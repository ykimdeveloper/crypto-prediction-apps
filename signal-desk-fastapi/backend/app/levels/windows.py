"""Per-zone stats over the last N fifteen-minute contract windows (aligned :00/:15/:30/:45).
A window counts as 'touched' when its 15m bar's high/low range overlaps the zone (price was in the zone at some point inside it).
Settlement uses app/settlement.py (tie -> UP) on the 15m bar: strike = bar open, end = bar close. That is an approximation of the
contract's index averages, not the exact strike, and the response says so (`settlement_basis`).
Held / broke / excursion come from the zone's touch log, for touches that STARTED inside the window."""
from __future__ import annotations

import numpy as np

from ..settlement import settles_up

W = 900_000
BASIS = "15m bar open -> close (approximation of the contract's 60s index averages); tie settles UP"


def window_grid(d15, as_of, n):
    """The last n COMPLETE windows before as_of, aligned to the quarter hour, and the 15m bars found for them (NaN where missing)."""
    starts = (as_of // W) * W - W * np.arange(n, 0, -1, dtype="int64")
    g = d15[d15["ts"].isin(starts)].drop_duplicates("ts").set_index("ts").reindex(starts)
    return starts, {k: g[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close")}


def zone_stats(zone, starts, bars, touches, want_table=False):
    covered = ~np.isnan(bars["close"])
    eligible = covered & (starts >= zone["scan_from"])
    with np.errstate(invalid="ignore"):
        touched = eligible & (bars["high"] >= zone["low"]) & (bars["low"] <= zone["high"])
        up = settles_up(bars["close"], bars["open"])
    by_win: dict[int, list] = {}
    for t in touches:
        by_win.setdefault(t["ts"] // W * W, []).append(t)
    inside = [t for s in starts[touched] for t in by_win.get(int(s), [])]
    cnt = lambda k: sum(1 for t in inside if t["outcome"] == k)
    exc = [t["excursion_atr"] for t in inside if t["excursion_atr"] is not None]
    out = dict(n_windows=int(len(starts)), windows_covered=int(covered.sum()), windows_eligible=int(eligible.sum()),
               windows_touched=int(touched.sum()), touched_settled_up=int((touched & up).sum()), touched_settled_down=int((touched & ~up).sum()),
               untouched_settled_up=int((eligible & ~touched & up).sum()), untouched_settled_down=int((eligible & ~touched & ~up).sum()),
               held=cnt("held"), broke=cnt("broke"), inconclusive=cnt("inconclusive"), pending=cnt("pending"),
               avg_excursion_atr=float(np.mean(exc)) if exc else None, settlement_basis=BASIS)
    if want_table:
        out["table"] = [dict(window_start=int(s), covered=bool(covered[i]), eligible=bool(eligible[i]), touched=bool(touched[i]),
                             settled=("UP" if up[i] else "DOWN") if covered[i] else None,
                             outcome=(by_win.get(int(s), [{}])[0].get("outcome") if touched[i] and by_win.get(int(s)) else None))
                        for i, s in enumerate(starts)]
    return out

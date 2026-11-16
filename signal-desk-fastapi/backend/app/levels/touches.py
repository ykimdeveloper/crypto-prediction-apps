"""The touch log. Touch = a candle's high/low enters the zone after the previous candle was fully outside it.
Outcome is measured over a forward horizon of H candles (default 15 minutes, at least one candle):
  broke         a candle closes beyond the zone (in the approach direction) and the next candle also closes beyond
  held          no close beyond the zone and price moved away >= x_atr * ATR from the zone edge (a "rejection")
  inconclusive  neither
  pending       the horizon (+1 confirming candle) is not fully closed yet; never counted in the stats
A `broke` outcome is a role flip: resistance became support (approach from below) or the reverse."""
from __future__ import annotations

import numpy as np


def horizon_bars(tf_ms: int, horizon_ms: int) -> int:
    return max(1, int(np.ceil(horizon_ms / tf_ms)))


def touch_log(d, zone, a, tf, tf_ms, horizon_ms=900_000, x_atr=0.5):
    ts, h, l, c = (d[k].to_numpy() for k in ("ts", "high", "low", "close"))
    av = a.to_numpy()
    lo, hi, n, H = zone["low"], zone["high"], len(d), horizon_bars(tf_ms, horizon_ms)
    start = max(1, int(np.searchsorted(ts, zone["scan_from"])))
    out, flips = [], []
    for i in range(start, n):
        if not (l[i] <= hi and h[i] >= lo) or (l[i - 1] <= hi and h[i - 1] >= lo):
            continue
        below = bool(h[i - 1] < lo)
        beyond = (lambda j: c[j] > hi) if below else (lambda j: c[j] < lo)
        t = dict(ts=int(ts[i]), tf=tf, approach="from_below" if below else "from_above", outcome="pending", excursion_atr=None)
        if i + H + 1 <= n - 1 and not np.isnan(av[i]) and av[i] > 0:
            win = range(i, i + H + 1)
            if any(beyond(j) and beyond(j + 1) for j in win):
                t["outcome"] = "broke"
                flips.append(dict(ts=int(ts[i]), from_role="resistance" if below else "support",
                                  to_role="support" if below else "resistance"))
            else:
                away = (lo - l[i:i + H + 1].min()) if below else (h[i:i + H + 1].max() - hi)
                t["excursion_atr"] = float(away / av[i])
                t["outcome"] = "held" if (away >= x_atr * av[i] and not any(beyond(j) for j in win)) else "inconclusive"
        out.append(t)
    return out, flips


def summarize(touches, flips, scan_from):
    cnt = lambda k: sum(1 for t in touches if t["outcome"] == k)
    done = cnt("held") + cnt("broke") + cnt("inconclusive")
    held, broke = cnt("held"), cnt("broke")
    return dict(n_touches=len(touches), held=held, broke=broke, inconclusive=cnt("inconclusive"), pending=cnt("pending"),
                first_seen=int(scan_from), last_touch=touches[-1]["ts"] if touches else None, n_role_flips=len(flips),
                evidence=f"held {held} of {done}" + (f", broke {broke}" if broke else "") if done else
                (f"{len(touches)} touch(es), none resolved yet" if touches else "no touches yet"))

"""Level candidates and zone clustering. Every candidate carries `scan_from`: the first time touches may be counted, so a
touch log never counts touches from before the level could have been known (pivot: when its last confirming bar closed;
prior period: when the period ended). Volume nodes and round numbers come from the whole span, so their touch history is
retrospective and flagged as such (the ML step must not train on those touches)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PIVOT_N = {60_000: 5, 300_000: 4, 900_000: 3, 1_800_000: 3, 3_600_000: 2}


def atr(d: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=max(3, n // 2)).mean()


def swing_pivots(d, a, tf_ms, min_prom_atr=1.0):
    n = PIVOT_N.get(tf_ms, 3)
    h, l, ts, av = d["high"].to_numpy(), d["low"].to_numpy(), d["ts"].to_numpy(), a.to_numpy()
    out = []
    for i in range(n, len(d) - n):          # a pivot at i needs n closed bars after it
        if np.isnan(av[i]):
            continue
        hw, lw = h[i - n:i + n + 1], l[i - n:i + n + 1]
        known = int(ts[i + n]) + tf_ms
        if hw.argmax() == n and h[i] - lw.min() >= min_prom_atr * av[i]:
            out.append(dict(price=float(h[i]), kind="pivot_high", ts=int(ts[i]), scan_from=known, retro=False))
        if lw.argmin() == n and hw.max() - l[i] >= min_prom_atr * av[i]:
            out.append(dict(price=float(l[i]), kind="pivot_low", ts=int(ts[i]), scan_from=known, retro=False))
    return out


def prior_period(d, as_of):
    """Previous COMPLETE day / week / month: open, high, low, close (UTC periods)."""
    out = []
    naive = pd.to_datetime(d["ts"], unit="ms")
    for code, name in (("D", "day"), ("W", "week"), ("M", "month")):
        per = naive.dt.to_period(code)
        done = [p for p in per.unique() if int((p + 1).start_time.tz_localize("UTC").timestamp() * 1000) <= as_of]
        if not done:
            continue
        p = max(done)
        g = d[(per == p).to_numpy()]
        end = int((p + 1).start_time.tz_localize("UTC").timestamp() * 1000)
        for k, v in (("open", g["open"].iloc[0]), ("high", g["high"].max()), ("low", g["low"].min()), ("close", g["close"].iloc[-1])):
            out.append(dict(price=float(v), kind=f"prev_{name}_{k}", ts=int(g["ts"].iloc[-1]), scan_from=end, retro=False))
    return out


def volume_nodes(d, a_last, bin_atr=0.25, peak_x=1.5):
    lo, hi = float(d["low"].min()), float(d["high"].max())
    step = bin_atr * a_last
    if not step > 0 or hi <= lo:
        return []
    nb = int(np.ceil((hi - lo) / step)) + 1
    idx = np.clip(((d[["high", "low", "close"]].mean(axis=1) - lo) / step).astype(int).to_numpy(), 0, nb - 1)
    prof = np.bincount(idx, weights=d["volume"].to_numpy(), minlength=nb)
    prof = np.convolve(prof, [0.25, 0.5, 0.25], mode="same")
    m = prof.mean()
    first = int(d["ts"].iloc[0])
    return [dict(price=float(lo + (i + 0.5) * step), kind="volume_node", ts=first, scan_from=first, retro=True)
            for i in range(1, nb - 1) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] >= peak_x * m]


def round_numbers(d, first_ts, limit=10):
    lo, hi = float(d["low"].min()), float(d["high"].max())
    step = 10 ** (int(np.floor(np.log10(max(hi, 1e-9)))) - 1)
    lv = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
    return [dict(price=float(p), kind="round_number", ts=first_ts, scan_from=first_ts, retro=True) for p in lv[:limit]]


def cluster(cands, width):
    """Merge candidates whose prices lie within `width` of each other (group span capped at 2*width)."""
    zones, cur = [], []
    for c in sorted(cands, key=lambda c: c["price"]):
        if cur and (c["price"] - cur[-1]["price"] > width or c["price"] - cur[0]["price"] > 2 * width):
            zones.append(cur)
            cur = []
        cur.append(c)
    if cur:
        zones.append(cur)
    out = []
    for g in zones:
        p = np.array([c["price"] for c in g])
        center = float(p.mean())
        out.append(dict(center=center, low=float(min(center - width / 2, p.min())), high=float(max(center + width / 2, p.max())),
                        sources=sorted({c["kind"] for c in g}), n_candidates=len(g), scan_from=min(c["scan_from"] for c in g),
                        retrospective=all(c["retro"] for c in g)))
    return out

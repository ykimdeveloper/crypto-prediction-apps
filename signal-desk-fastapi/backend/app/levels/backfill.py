"""Data policy for the levels grid (spec section 3) and a coverage report.

  1m            recent weeks only (capped at 45 days here); older spans resample from native 5m/15m.
  5m/15m/30m/1h backfilled natively through CCXT for up to two years.

This reuses quant15.bars.backfill (public REST endpoints, no key, idempotent upserts, per-exchange page caps) and never touches
quant15 or the desk. It only fetches history into the `candles` table. The coverage report says, per timeframe and per
(timeframe, span) cell, where the stored candles start and end, where the gaps are, and whether the cell would be computed
or reported as short_history / gapped / not_enough_structure (the same thresholds the levels service uses)."""
from __future__ import annotations

import threading
import time

import numpy as np

from . import service
from .data import DAY, SPAN_MS, TF_MS

NATIVE_TFS = ("5m", "15m", "30m", "1h")
MAX_1M_DAYS = 45
MAX_GAPS_LISTED = 5
GAP_SCAN_1M_DAYS = 30           # a gap scan over two years of 1m rows is not worth its cost
_lock = threading.Lock()
STATE: dict = dict(state="idle")


def run(exchange, symbols, tfs=NATIVE_TFS, days=730, backfill_fn=None):
    if not _lock.acquire(blocking=False):
        raise RuntimeError("a levels backfill is already running")
    try:
        if backfill_fn is None:
            from ..quant15.bars import backfill as backfill_fn
        STATE.clear()
        STATE.update(state="running", started=int(time.time()), exchange=exchange, results=[])
        for s in symbols:
            for tf in tfs:
                d = min(days, MAX_1M_DAYS) if tf == "1m" else days
                STATE["current"] = f"{s} {tf}"
                try:
                    res = dict(backfill_fn(exchange, s, d, tf), days=d)
                    if tf == "1m" and d < days:
                        res["note"] = f"1m capped at {MAX_1M_DAYS} days by policy; older spans use native 5m/15m"
                except Exception as e:  # noqa: BLE001  one failing symbol/timeframe must not hide the others
                    res = dict(symbol=s, tf=tf, error=f"{type(e).__name__}: {str(e)[:200]}")
                STATE["results"].append(res)
        errs = [r for r in STATE["results"] if r.get("error")]
        STATE.update(state="done_with_errors" if errs else "done", finished=int(time.time()), current=None)
        return STATE
    finally:
        _lock.release()


def status():
    return dict(STATE)


def db_timestamps(exchange, symbol, tf, since=None):
    from .. import db
    q, a = "SELECT ts FROM candles WHERE exchange=? AND symbol=? AND tf=?", [exchange, symbol, tf]
    if since is not None:
        q, a = q + " AND ts>=?", a + [since]
    with db._lock:
        rows = db._conn.execute(q + " ORDER BY ts", a).fetchall()
    return np.array([r[0] for r in rows], dtype="int64")


def _gaps(ts, tf_ms):
    d = np.diff(ts)
    idx = np.nonzero(d > 1.5 * tf_ms)[0]
    gaps = [dict(after_ts=int(ts[i]), missing_candles=int(d[i] // tf_ms - 1)) for i in idx]
    return len(gaps), sorted(gaps, key=lambda g: -g["missing_candles"])[:MAX_GAPS_LISTED]


def _cell(ts, tf_ms, span_ms, now):
    """Same gate as service.gate, evaluated from timestamps only (no candle values needed)."""
    inside = ts[(ts + tf_ms <= now) & (ts >= now - span_ms)]
    if len(inside) < service.MIN_CANDLES:
        return dict(status="not_enough_structure", n_candles=int(len(inside)), note=f"{len(inside)} candles; need at least {service.MIN_CANDLES}")
    first, last = int(inside[0]), int(inside[-1])
    expected = (last - first) // tf_ms + 1
    covered = min(1.0, (last + tf_ms - first) / span_ms)
    missing = 1 - len(inside) / expected
    if covered < service.MIN_COVERED_FRAC:
        return dict(status="short_history", n_candles=int(len(inside)), covered_frac=float(covered), note=f"stored candles cover {covered:.0%} of the span")
    if missing > service.MAX_MISSING_FRAC:
        return dict(status="gapped", n_candles=int(len(inside)), missing_frac=float(missing), note=f"{missing:.0%} of the candles are missing")
    return dict(status="ok", n_candles=int(len(inside)), covered_frac=float(covered), missing_frac=float(max(0.0, missing)), heavy=len(inside) > service.HEAVY_CANDLES)


def coverage(exchange, symbol, ts_loader=None, now_ms=None):
    ts_loader = ts_loader or db_timestamps
    now = int(now_ms if now_ms is not None else time.time() * 1000)
    tfs, grid = [], []
    for tf, tf_ms in TF_MS.items():
        ts = np.asarray(ts_loader(exchange, symbol, tf), dtype="int64")
        if len(ts) == 0:
            tfs.append(dict(tf=tf, n=0, status="no_candles"))
            for sp in SPAN_MS:
                if sp != "1y":
                    grid.append(dict(tf=tf, span=sp, status="not_enough_structure", n_candles=0, note="no stored candles"))
            continue
        scan = ts[ts >= now - GAP_SCAN_1M_DAYS * DAY] if tf == "1m" else ts
        n_gaps, worst = _gaps(scan, tf_ms) if len(scan) > 1 else (0, [])
        expected = int((ts[-1] - ts[0]) // tf_ms + 1)
        tfs.append(dict(tf=tf, n=int(len(ts)), first=int(ts[0]), last=int(ts[-1]), days=float((ts[-1] - ts[0]) / DAY), last_age_hours=float((now - ts[-1]) / 3.6e6),
                        expected=expected, missing_frac=float(max(0.0, 1 - len(ts) / expected)), gaps=n_gaps, worst_gaps=worst,
                        gap_scan=f"last {GAP_SCAN_1M_DAYS} days only" if tf == "1m" else "full history"))
        for sp, ms in SPAN_MS.items():
            if sp != "1y":
                grid.append(dict(tf=tf, span=sp, **_cell(ts, tf_ms, ms, now)))
    return dict(exchange=exchange, symbol=symbol, as_of=now, timeframes=tfs, cells=grid,
                policy=f"1m: recent weeks only (<= {MAX_1M_DAYS} days). 5m/15m/30m/1h: native, up to 2 years. A cell is computed only when status is ok.")

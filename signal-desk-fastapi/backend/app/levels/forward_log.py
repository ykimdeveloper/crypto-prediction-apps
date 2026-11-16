"""Forward journal: the out-of-sample evidence no backtest can fake.

At every 15-minute window start it records what the toolkit showed (levels, nearest distances, confluence, the settings in force).
When the window has closed it records what price did around each level and how the window settled.

Rules that keep it honest:
  * the snapshot is computed with as_of = window start, from candles closed before it, and is written only within LATE_MS of that start
    (a process that was down is never back-filled: a back-filled row would not be forward evidence);
  * it runs only when Levels is On (journal defaults to on); with the toggle off nothing is computed and nothing is written;
  * it lives in its own asyncio task with its own SQLite connection. It never touches hub, strategy, paper or the feeds.
Settlement uses app/settlement.py on the window's 1m candles (open of the first -> close of the last). That is an approximation of the
contract's index averages and is stored in `settle_basis`."""
from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
import time

from .. import config as app_config
from ..settlement import settle_dir
from . import config as levels_config, confluence as CF, service

W = 900_000
DELAY_MS = 5_000            # wake this long after the boundary so the previous 1m candle is stored
LATE_MS = 120_000           # a snapshot later than this after the window start is skipped, not back-filled
SETTLE_DELAY_MS = 20_000
GIVE_UP_MS = 30 * 60_000    # unsettled this long after the window end -> status 'no_data'
BASIS = "1m candles of the window: first open -> last close (approximation of the 60s index averages); tie settles UP"
SCHEMA = """
CREATE TABLE IF NOT EXISTS levels_journal_windows(
  id INTEGER PRIMARY KEY, exchange TEXT NOT NULL, symbol TEXT NOT NULL, window_start INTEGER NOT NULL, as_of INTEGER, recorded_ms INTEGER,
  settings TEXT, close REAL, n_zones INTEGER, nearest TEXT, confluence TEXT, cell_status TEXT,
  status TEXT NOT NULL DEFAULT 'open', win_open REAL, win_high REAL, win_low REAL, win_close REAL, settled TEXT, settle_basis TEXT, settled_ms INTEGER,
  UNIQUE(exchange, symbol, window_start));
CREATE TABLE IF NOT EXISTS levels_journal_zones(
  window_id INTEGER NOT NULL, zone_id TEXT NOT NULL, cell TEXT, low REAL, high REAL, center REAL, side TEXT, dist_bp REAL, dist_atr REAL,
  n_touches INTEGER, held INTEGER, broke INTEGER, edge TEXT, confluence_count INTEGER,
  touched INTEGER, first_touch_ts INTEGER, max_above_bp REAL, max_below_bp REAL, close_side TEXT,
  PRIMARY KEY(window_id, zone_id));
"""
_conn = None
_lock = threading.Lock()


def connect(path=None):
    c = sqlite3.connect(path or app_config.DB_PATH, check_same_thread=False, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.executescript(SCHEMA)
    return c


def _shared():
    global _conn
    with _lock:
        if _conn is None:
            _conn = connect()
        return _conn


def discover_targets(conn, now_ms):
    """(exchange, symbol) pairs whose feed wrote a 1m candle in the last hour. Read-only."""
    try:
        rows = conn.execute("SELECT DISTINCT exchange, symbol FROM candles WHERE tf='1m' AND ts > ?", (now_ms - 3_600_000,)).fetchall()
    except sqlite3.Error:
        return []
    return [(r[0], r[1]) for r in rows]


def snapshot(conn, exchange, symbol, window_start, cfg, now_ms, loader=None):
    if conn.execute("SELECT 1 FROM levels_journal_windows WHERE exchange=? AND symbol=? AND window_start=?", (exchange, symbol, window_start)).fetchone():
        return "exists"
    if now_ms - window_start > LATE_MS:
        return "late"
    res = service.levels_for(exchange, symbol, cfg, as_of=window_start, loader=loader)
    ok = [c for c in res["cells"] if c["status"] == "ok"]
    conf = CF.merge(res["cells"])
    count = {i: g["confluence_count"] for g in conf for i in g["zone_ids"]}
    cur = conn.execute("INSERT INTO levels_journal_windows(exchange,symbol,window_start,as_of,recorded_ms,settings,close,n_zones,nearest,confluence,cell_status)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (exchange, symbol, window_start, window_start, now_ms, json.dumps(cfg), ok[0]["close"] if ok else None,
                        sum(len(c["zones"]) for c in ok),
                        json.dumps({f"{c['tf']}/{c['span']}": [c.get("nearest_above"), c.get("nearest_below")] for c in ok}),
                        json.dumps([{k: g[k] for k in ("id", "low", "high", "cells", "confluence_count", "score")} for g in conf]),
                        json.dumps({f"{c['tf']}/{c['span']}": c["status"] for c in res["cells"]})))
    wid = cur.lastrowid
    for c in ok:
        for z in c["zones"]:
            conn.execute("INSERT OR IGNORE INTO levels_journal_zones(window_id,zone_id,cell,low,high,center,side,dist_bp,dist_atr,n_touches,held,broke,edge,confluence_count)"
                         " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (wid, z["id"], z["cell"], z["low"], z["high"], z["center"], z["side"], z["distance"]["bp"], z["distance"]["atr"],
                          z["n_touches"], z["held"], z["broke"], z["edge"], count.get(z["id"], 1)))
    conn.commit()
    return "written"


def settle_pending(conn, now_ms, loader=None):
    if loader is None:
        from .data import load as loader
    rows = conn.execute("SELECT * FROM levels_journal_windows WHERE status='open' AND window_start + ? + ? <= ?", (W, SETTLE_DELAY_MS, now_ms)).fetchall()
    frames, n = {}, 0
    for r in rows:
        key = (r["exchange"], r["symbol"])
        if key not in frames:
            frames[key] = loader(key[0], key[1], "1m")
        d = frames[key]
        s, e = r["window_start"], r["window_start"] + W
        g = d[(d["ts"] >= s) & (d["ts"] < e)] if len(d) else d
        if len(g) < 12 or int(g["ts"].iloc[0]) > s + 60_000:
            if now_ms - e > GIVE_UP_MS:
                conn.execute("UPDATE levels_journal_windows SET status='no_data', settled_ms=? WHERE id=?", (now_ms, r["id"]))
            continue
        o, c, hi, lo = float(g["open"].iloc[0]), float(g["close"].iloc[-1]), float(g["high"].max()), float(g["low"].min())
        conn.execute("UPDATE levels_journal_windows SET status='settled', win_open=?, win_high=?, win_low=?, win_close=?, settled=?, settle_basis=?, settled_ms=? WHERE id=?",
                     (o, hi, lo, c, settle_dir(c, o), BASIS, now_ms, r["id"]))
        px = r["close"] or o
        for z in conn.execute("SELECT * FROM levels_journal_zones WHERE window_id=?", (r["id"],)).fetchall():
            hit = g[(g["low"] <= z["high"]) & (g["high"] >= z["low"])]
            conn.execute("UPDATE levels_journal_zones SET touched=?, first_touch_ts=?, max_above_bp=?, max_below_bp=?, close_side=? WHERE window_id=? AND zone_id=?",
                         (int(len(hit) > 0), int(hit["ts"].iloc[0]) if len(hit) else None, (hi - z["high"]) / px * 1e4, (z["low"] - lo) / px * 1e4,
                          "above" if c > z["high"] else "below" if c < z["low"] else "inside", r["id"], z["zone_id"]))
        n += 1
    conn.commit()
    return n


def tick(now_ms, conn=None, cfg=None, loader=None, targets=None):
    """One journal cycle. Returns {} without touching anything when Levels is Off or the journal is disabled."""
    cfg = cfg or levels_config.get()
    if not cfg["enabled"] or not cfg.get("journal", True):
        return {}
    conn = conn or _shared()
    ws = now_ms // W * W
    out = {}
    for ex, sym in (targets if targets is not None else discover_targets(conn, now_ms)):
        try:
            out[f"{ex}:{sym}"] = snapshot(conn, ex, sym, ws, cfg, now_ms, loader)
        except Exception as e:      # one bad symbol must not stop the others
            out[f"{ex}:{sym}"] = f"error: {type(e).__name__}: {str(e)[:120]}"
    out["settled"] = settle_pending(conn, now_ms, loader)
    return out


def wilson(k, n, z=1.645):
    """90% Wilson interval for a proportion."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def summary(conn, exchange, symbol):
    rows = conn.execute("""SELECT w.id, w.settled, MAX(COALESCE(z.touched, 0)) AS touched, MAX(CASE WHEN z.confluence_count >= 2 THEN COALESCE(z.touched, 0) ELSE 0 END) AS touched_conf
                           FROM levels_journal_windows w LEFT JOIN levels_journal_zones z ON z.window_id = w.id
                           WHERE w.exchange=? AND w.symbol=? AND w.status='settled' GROUP BY w.id""", (exchange, symbol)).fetchall()

    def grp(pred):
        sel = [r for r in rows if pred(r)]
        up = sum(1 for r in sel if r["settled"] == "UP")
        return dict(windows=len(sel), settled_up=up, up_rate=(up / len(sel) if sel else None), up_rate_interval90=wilson(up, len(sel)))
    return dict(windows_settled=len(rows), any_zone_touched=grp(lambda r: r["touched"]), no_zone_touched=grp(lambda r: not r["touched"]),
                confluence_zone_touched=grp(lambda r: r["touched_conf"]), settle_basis=BASIS,
                note="Descriptive only: no significance test, no baseline comparison yet, and small samples give wide intervals.")


def recent(conn, exchange, symbol, limit=50):
    ws = conn.execute("SELECT * FROM levels_journal_windows WHERE exchange=? AND symbol=? ORDER BY window_start DESC LIMIT ?", (exchange, symbol, limit)).fetchall()
    out = []
    for w in ws:
        zs = [dict(z) for z in conn.execute("SELECT * FROM levels_journal_zones WHERE window_id=? ORDER BY center", (w["id"],)).fetchall()]
        d = dict(w)
        for k in ("settings", "nearest", "confluence", "cell_status"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append({**d, "zones": zs})
    return out


async def _run():
    while True:
        now = int(time.time() * 1000)
        await asyncio.sleep(max(1.0, ((now // W + 1) * W + DELAY_MS - now) / 1000))
        try:
            await asyncio.to_thread(tick, int(time.time() * 1000))
        except Exception as e:
            print(f"[levels] journal tick failed: {type(e).__name__}: {e}")


def start():
    return asyncio.get_running_loop().create_task(_run())

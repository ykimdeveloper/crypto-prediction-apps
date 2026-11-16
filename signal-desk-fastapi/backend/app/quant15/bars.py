"""Bars: CCXT backfill -> SQLite, and 1m -> 15m resampling aligned to the contract windows."""
from __future__ import annotations

import pandas as pd

from .. import db
from ..approaches.features import to_frame
from ..constants import CONTRACT_MS, MIN_MS

BAR_MS = CONTRACT_MS  # one 15m bar == one contract window
COLS = ["ts", "open", "high", "low", "close", "volume"]


def resample_15m(d1: pd.DataFrame, min_minutes: int = 13) -> pd.DataFrame:
    """1m OHLCV (ts = candle open, epoch ms, sorted) -> 15m bars keyed by window start. Windows with fewer than
    `min_minutes` of 15 minute-candles are dropped rather than guessed."""
    if d1.empty:
        return pd.DataFrame(columns=COLS)
    key = (d1["ts"].astype("int64") // BAR_MS) * BAR_MS
    out = d1.groupby(key).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                              close=("close", "last"), volume=("volume", "sum"), n=("ts", "size"))
    out = out[out["n"] >= min_minutes].drop(columns="n")
    out.index.name = "ts"
    return out.reset_index()


def closed_only(bars: pd.DataFrame, now_ms: int) -> pd.DataFrame:
    return bars[bars["ts"] + BAR_MS <= now_ms].reset_index(drop=True)


def load_1m(exchange: str, symbol: str) -> pd.DataFrame:
    return to_frame(db.dataset(exchange, symbol, "1m"))


def load_15m(exchange: str, symbol: str, now_ms: int | None = None) -> pd.DataFrame:
    """Native 15m rows (from `backfill(..., tf='15m')`) first, then windows derived from stored 1m candles for anything else."""
    native = to_frame(db.dataset(exchange, symbol, "15m"))
    derived = resample_15m(load_1m(exchange, symbol))
    parts = [p for p in (native, derived) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=COLS)
    bars = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="first").sort_values("ts").reset_index(drop=True)
    bars["ts"] = bars["ts"].astype("int64")
    return closed_only(bars, now_ms) if now_ms is not None else bars


# Max candles one REST call returns. Asking for more makes some exchanges error out (Coinbase rejects > 350) or silently truncate.
PAGE_CAP = {"coinbase": 300, "coinbaseexchange": 300, "okx": 100, "kraken": 720}


def backfill(exchange: str, symbol: str, days: int = 30, tf: str = "1m", ex=None, page: int = 1000) -> dict:
    """Pull history through CCXT REST (public endpoints, no key) into `candles`.
    Resumes after the newest stored candle ONLY when the stored candles already reach back to the requested start; otherwise it
    re-fetches from the start (upserts are idempotent). Without that check a few hours recorded by the live feed made a 30-day
    backfill fetch nothing but the last few minutes.
    `ex` lets tests inject a fake exchange. Use tf='1m' for ~30-90 days (what the replay needs) or tf='15m' for years of model history."""
    if ex is None:
        import ccxt  # imported lazily so the rest of the package works without it
        ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    page = min(page, PAGE_CAP.get(exchange, page))
    tf_ms = int(ex.parse_timeframe(tf) * 1000)
    end = int(ex.milliseconds())
    start = end - int(days) * 86_400_000
    since = start
    last = db.query("candles", exchange=exchange, symbol=symbol, tf=tf, limit=1, order="DESC")
    first = db.query("candles", exchange=exchange, symbol=symbol, tf=tf, limit=1, order="ASC")
    if last and first and int(first[0]["ts"]) <= start + 2 * tf_ms:
        since = max(since, int(last[0]["ts"]) + tf_ms)
    total, first_ts = 0, None
    while since < end:
        rows = ex.fetch_ohlcv(symbol, tf, since=since, limit=page)
        if not rows:          # a quiet stretch on a thin pair is an empty page, not the end of history: skip past it
            since += page * tf_ms
            continue
        candles = [dict(t=int(r[0]), o=r[1], h=r[2], l=r[3], c=r[4], v=r[5] or 0.0) for r in rows if None not in r[:5]]
        db.upsert_candles(exchange, symbol, tf, candles)
        total += len(candles)
        first_ts = first_ts if first_ts is not None else int(rows[0][0])
        nxt = int(rows[-1][0]) + tf_ms
        if nxt <= since:  # exchange ignored `since`: stop rather than loop forever
            break
        since = nxt
    return dict(exchange=exchange, symbol=symbol, tf=tf, fetched=total, from_ts=first_ts, to_ts=since - tf_ms)

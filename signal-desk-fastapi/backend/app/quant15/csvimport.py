"""Load 1-minute OHLCV CSVs (CryptoDataDownload, or Binance Vision klines) into the SQLite `candles` table.

    cd backend
    python -m app.quant15.csvimport --exchange binance --symbol BTC/USDT --days 60 Binance_BTCUSDT_minute.csv
    python -m app.quant15.csvimport --exchange binance --symbol ETH/USDT --days 60 Binance_ETHUSDT_minute.csv

Handles: the one-line banner CDD puts above the header, newest-first ordering, ms / microsecond / second timestamps,
headerless Binance Vision files (open_time,o,h,l,c,volume,...), and .zip files. Reports gaps instead of filling them.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .. import db
from ..constants import MIN_MS

COLS = ["ts", "o", "h", "l", "c", "v"]


def _header_row(path: str) -> int | None:
    """Index of the header line (the one naming 'open'), or None for headerless files."""
    with open(path, "rt", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 5:
                break
            low = line.lower()
            if "open" in low and "close" in low:
                return i
    return None


def _to_ms(ts: pd.Series) -> pd.Series:
    """Epoch seconds / ms / microseconds -> ms, floored to the minute (candle open)."""
    ts = pd.to_numeric(ts, errors="coerce").dropna().astype("int64")
    m = float(ts.median()) if len(ts) else 0.0
    if m > 1e14:
        ts = ts // 1000        # microseconds (Binance spot files from 2025)
    elif m < 1e11:
        ts = ts * 1000         # seconds
    return ts - ts % MIN_MS


def read_1m(path: str, symbol: str | None = None, days: int | None = None) -> pd.DataFrame:
    """Return ascending 1m candles [ts,o,h,l,c,v]; `days` keeps the last N days counted back from the newest row in the file."""
    hdr = _header_row(path)
    if hdr is None:  # Binance Vision: no header, positional
        raw = pd.read_csv(path, header=None, usecols=range(6), names=COLS)
        raw["ts"] = _to_ms(raw["ts"])
        d = raw
    else:
        peek = pd.read_csv(path, skiprows=hdr, nrows=2)
        low = {c.lower().strip(): c for c in peek.columns}
        need = ["unix", "open", "high", "low", "close"]
        if any(k not in low for k in need):
            raise ValueError(f"{path}: expected columns Unix/Open/High/Low/Close, got {list(peek.columns)}")
        nrows = None
        if days and len(peek) == 2:  # CDD files run newest-first: read only the top of a multi-year file
            t = _to_ms(peek[low["unix"]])
            if len(t) == 2 and t.iloc[0] > t.iloc[1]:
                nrows = int(days) * 1440 + 5000
        raw = pd.read_csv(path, skiprows=hdr, nrows=nrows)
        base = (symbol or "").split("/")[0].lower()
        vol = next((low[k] for k in low if k.startswith("volume") and base and k.split()[-1] == base), None) \
            or next((low[k] for k in low if k.startswith("volume")), None)
        if vol is None:
            raise ValueError(f"{path}: no Volume column")
        d = pd.DataFrame({"ts": _to_ms(raw[low["unix"]]), "o": raw[low["open"]], "h": raw[low["high"]],
                          "l": raw[low["low"]], "c": raw[low["close"]], "v": raw[vol]})
    d = d.dropna().drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    d["ts"] = d["ts"].astype("int64")
    if days and len(d):
        d = d[d["ts"] > d["ts"].iloc[-1] - int(days) * 86_400_000].reset_index(drop=True)
    return d


def gaps(d: pd.DataFrame) -> dict:
    if len(d) < 2:
        return dict(missing_minutes=0, largest_gap_min=0)
    step = np.diff(d["ts"].to_numpy()) // MIN_MS
    return dict(missing_minutes=int((step - 1).clip(min=0).sum()), largest_gap_min=int(step.max() - 1))


def import_csv(paths: list[str], exchange: str, symbol: str, days: int | None = None) -> dict:
    frames = [read_1m(p, symbol, days) for p in paths]
    d = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    if days and len(d):
        d = d[d["ts"] > d["ts"].iloc[-1] - int(days) * 86_400_000].reset_index(drop=True)
    if d.empty:
        raise ValueError("no rows to import")
    for i in range(0, len(d), 50_000):
        chunk = d.iloc[i:i + 50_000]
        db.upsert_candles(exchange, symbol, "1m", [dict(t=int(r.ts), o=r.o, h=r.h, l=r.l, c=r.c, v=float(r.v))
                                                   for r in chunk.itertuples(index=False)])
    return dict(exchange=exchange, symbol=symbol, tf="1m", rows=len(d), from_ts=int(d["ts"].iloc[0]),
                to_ts=int(d["ts"].iloc[-1]), **gaps(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exchange", required=True, help="label stored in SQLite; use the real venue, e.g. binance")
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT (same format the live feed writes)")
    ap.add_argument("--days", type=int, default=90, help="keep the newest N days of the file(s)")
    a = ap.parse_args(argv)
    db.init()
    out = import_csv(a.files, a.exchange, a.symbol, a.days)
    print(out)
    if out["missing_minutes"]:
        print(f"note: {out['missing_minutes']} missing minutes (largest gap {out['largest_gap_min']} min); windows with holes are dropped, not guessed")


if __name__ == "__main__":
    main()

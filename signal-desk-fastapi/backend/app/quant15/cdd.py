"""Fetch 1m history from CryptoDataDownload when the database doesn't hold enough for the 15m model.

CDD publishes Binance SPOT minute files, so this only serves the exchange label `binance`; any other exchange keeps using CCXT.
The file is one multi-year CSV (hundreds of MB), downloaded once and cached under backend/data/cdd for 12 hours; only the newest
`days` are imported. The file lags real time by about a day, so `bars.backfill` (CCXT) can top up the last stretch.
Override the address with CDD_URL_TEMPLATE, e.g. https://www.cryptodatadownload.com/cdd/Binance_{base}{quote}_minute.csv
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .. import config
from . import bars as B, csvimport, model15 as M

SUPPORTED = {"binance"}
URL = os.getenv("CDD_URL_TEMPLATE", "https://www.cryptodatadownload.com/cdd/Binance_{base}{quote}_minute.csv")
CACHE = Path(config.ROOT) / "data" / "cdd"
MAX_AGE_S = 12 * 3600
MIN_WINDOWS = M.MIN_BARS + 100      # 15m windows a symbol needs before compare/train can run


def url_for(symbol: str) -> str:
    base, _, quote = symbol.partition("/")
    return URL.format(base=base.upper(), quote=(quote or "USDT").upper())


def download(symbol: str, force: bool = False, http=requests) -> Path:
    url = url_for(symbol)
    path = CACHE / url.rsplit("/", 1)[-1]
    if not force and path.exists() and time.time() - path.stat().st_mtime < MAX_AGE_S:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    r = http.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=(15, 120))
    r.raise_for_status()
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with open(tmp, "rb") as f:
        head = f.read(2000).lower()
    if b"<html" in head or b"open" not in head:      # a login wall or moved file, not a CSV
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{url} did not return a CSV (login wall or moved file). Download it by hand and run "
                           f"`python -m app.quant15.csvimport --exchange binance --symbol {symbol} FILE.csv`")
    tmp.replace(path)
    return path


def windows(exchange: str, symbol: str) -> int:
    return len(B.load_15m(exchange, symbol))


def fetch_into(symbol: str, exchange: str = "binance", days: int = 90, force: bool = False, http=requests) -> dict:
    """Always download (cache permitting) and import the newest `days`."""
    if exchange not in SUPPORTED:
        raise ValueError(f"CryptoDataDownload only serves Binance spot; '{exchange}' must use CCXT (POST /backfill with source=ccxt)")
    path = download(symbol, force=force, http=http)
    res = csvimport.import_csv([str(path)], exchange, symbol, days)
    age_days = round((time.time() * 1000 - res["to_ts"]) / 86_400_000, 1)
    out = dict(source="cdd", windows=windows(exchange, symbol), newest_age_days=age_days, **res)
    if age_days > 3:
        out["warning"] = (f"the CryptoDataDownload file ends {age_days} days ago: fine for training, but the live model view needs "
                          f"~8 days of CONTIGUOUS recent 1m candles for this exchange (POST /backfill with source=ccxt, or let the feed run)")
    return out


def ensure(symbol: str, exchange: str = "binance", days: int = 90, min_windows: int = MIN_WINDOWS, http=requests) -> dict:
    """Only fetch when the database is short of history."""
    have = windows(exchange, symbol)
    if have >= min_windows:
        return dict(symbol=symbol, exchange=exchange, source="db", fetched=False, windows=have)
    return dict(fetched=True, **fetch_into(symbol, exchange, days, http=http))

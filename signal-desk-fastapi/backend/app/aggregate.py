"""Consolidate the same base asset across several CCXT exchanges into ONE candle series.

The consolidated series is fed to the exact same Engine (ta Bollinger / StochRSI / ATR + the three strategies) that a single
exchange uses, so "use every exchange's data" is just "give the engine better candles". Everything here is a pure function.

Assumption worth knowing: USDT / USD / USDC quotes are treated as one dollar (config.AGG_QUOTES). Exchanges that only list the
asset against KRW, AUD, EUR ... are skipped, because merging prices in different currencies would corrupt the indicators.
"""
from __future__ import annotations

from statistics import median

from . import config

AGG_PREFIX = "agg:"


def agg_id(exchanges: list[str]) -> str:
    """Stable id for a set of exchanges (order-independent). Used as the 'exchange' key in SQLite and the paper ledger."""
    return AGG_PREFIX + "+".join(sorted(set(exchanges)))


def is_agg(exchange: str) -> bool:
    return exchange.startswith(AGG_PREFIX)


def members(aid: str) -> list[str]:
    return [x for x in aid[len(AGG_PREFIX):].split("+") if x]


def pick_symbol(markets: dict, base: str, quotes=None) -> str | None:
    """First listed, active SPOT market for `base` against the preferred quotes (in order). None if there isn't one."""
    for q in quotes or config.AGG_QUOTES:
        sym = f"{base}/{q}"
        m = markets.get(sym)
        if m is None:
            continue
        if m.get("active") is False:
            continue
        if m.get("spot") is False:  # None (unknown) is accepted; explicit False (swap/future/margin-only) is not
            continue
        return sym
    return None


def _weights(rows: list[dict]) -> list[float]:
    vols = [max(r["v"], 0.0) for r in rows]
    tot = sum(vols)
    return [v / tot for v in vols] if tot > 0 else [1.0 / len(rows)] * len(rows)


def merge_one(rows: list[dict], mode: str = "vwap") -> dict:
    """Merge the same minute from several venues. vwap = volume-weighted O/H/L/C; median = per-field median."""
    if len(rows) == 1:
        r = rows[0]
        return dict(t=r["t"], o=r["o"], h=r["h"], l=r["l"], c=r["c"], v=r["v"], n=1)
    if mode == "median":
        o, h, l, c = (median(r[k] for r in rows) for k in ("o", "h", "l", "c"))
    else:
        w = _weights(rows)
        o, h, l, c = (sum(wi * r[k] for wi, r in zip(w, rows)) for k in ("o", "h", "l", "c"))
    h, l = max(h, o, c), min(l, o, c)  # a merged candle must still be a valid candle
    return dict(t=rows[0]["t"], o=o, h=h, l=l, c=c, v=sum(r["v"] for r in rows), n=len(rows))


def merge_candles(per_venue: dict[str, dict[int, dict]], mode: str | None = None, limit: int = 400,
                  max_dev: float | None = None) -> tuple[list[dict], dict[str, dict]]:
    """per_venue = {exchange: {t: candle}}  ->  (consolidated candles oldest-first, per-venue stats).

    Outlier guard: with 3+ venues reporting a minute, a venue whose close is more than `max_dev` (fraction) away from the median
    close is left out of that minute (a depegged stablecoin, a stale illiquid book, a bad market mapping ...).
    The stats say how many minutes each venue contributed / was rejected, so the UI can show it.
    """
    mode = mode or config.AGG_MODE
    max_dev = config.AGG_MAX_DEV if max_dev is None else max_dev
    stats = {ex: dict(used=0, rejected=0) for ex in per_venue}
    ts = sorted({t for rows in per_venue.values() for t in rows})[-limit:]
    out = []
    for t in ts:
        got = [(ex, rows[t]) for ex, rows in per_venue.items() if t in rows]
        if len(got) >= 3 and max_dev:
            mid = median(r["c"] for _, r in got)
            kept = [(ex, r) for ex, r in got if mid and abs(r["c"] - mid) / mid <= max_dev]
            for ex, _ in got:
                stats[ex]["rejected" if all(ex != k for k, _ in kept) else "used"] += 1
            got = kept
        else:
            for ex, _ in got:
                stats[ex]["used"] += 1
        if got:
            out.append(merge_one([r for _, r in got], mode))
    return out, stats


def best_book(tickers: dict[str, dict | None], now: int, max_age_ms: int = 90_000) -> dict | None:
    """Consolidated best bid / best ask across venues (highest bid, lowest ask). Crossed books are kept as-is: that IS the arb."""
    bids, asks, lasts = [], [], []
    for t in tickers.values():
        if not t or now - (t.get("ts") or 0) > max_age_ms:
            continue
        if t.get("bid") is not None and t.get("ask") is not None:
            bids.append(t["bid"])
            asks.append(t["ask"])
        if t.get("last") is not None:
            lasts.append(t["last"])
    if not bids:
        return None
    return dict(ts=now, last=median(lasts) if lasts else None, bid=max(bids), ask=min(asks))

"""Check what an exchange will actually give us over public CCXT REST, before training on it.

    python -m app.quant15.probe --exchange coinbase --symbol BTC/USDT
    python -m app.quant15.probe --exchange coinbaseexchange --symbol BTC/USD

Prints: the ccxt version, whether the symbol is listed, and the result of a real 1m fetch from ~3 days ago (count, first/last
timestamp, or the exact exception). Also tries the common alternatives (other exchange id, USD/USDC quotes) so you can see
which label/symbol pair works.
"""
from __future__ import annotations

import argparse
import time


def try_fetch(ex, symbol: str, days: float = 3) -> str:
    since = int(ex.milliseconds() - days * 86_400_000)
    try:
        rows = ex.fetch_ohlcv(symbol, "1m", since=since, limit=300)
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {str(e)[:300]}"
    if not rows:
        return "0 rows returned"
    f = lambda t: time.strftime("%F %T", time.gmtime(t / 1000))
    return f"{len(rows)} rows, {f(rows[0][0])} -> {f(rows[-1][0])} UTC (asked for since {f(since)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="coinbase")
    ap.add_argument("--symbol", default="BTC/USDT")
    a = ap.parse_args()
    import ccxt
    print("ccxt", ccxt.__version__)
    base, _, quote = a.symbol.partition("/")
    combos = [(a.exchange, a.symbol)]
    for ex_id in dict.fromkeys([a.exchange, "coinbase", "coinbaseexchange"]):
        for q in dict.fromkeys([quote or "USDT", "USD", "USDC", "USDT"]):
            if (ex_id, f"{base}/{q}") not in combos:
                combos.append((ex_id, f"{base}/{q}"))
    clients: dict = {}
    for ex_id, sym in combos:
        try:
            ex = clients.get(ex_id) or clients.setdefault(ex_id, getattr(ccxt, ex_id)({"enableRateLimit": True}))
            if not ex.markets:
                ex.load_markets()
        except Exception as e:  # noqa: BLE001
            print(f"{ex_id:17s} {sym:10s} load_markets failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        if sym not in ex.markets:
            print(f"{ex_id:17s} {sym:10s} not listed")
            continue
        print(f"{ex_id:17s} {sym:10s} {try_fetch(ex, sym)}")


if __name__ == "__main__":
    main()

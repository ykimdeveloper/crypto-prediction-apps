"""Multi-timeframe Coinbase analyzer -> text report -> Claude -> buy/wait/avoid alerts."""
import asyncio, math, os, time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import ccxt
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

import contract
import quant

load_dotenv()
SYMBOL = os.getenv("SYMBOL", "BTC/USD")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
LOOKBACK_MIN = int(os.getenv("LOOKBACK_MINUTES", "10080"))  # 7 days of 1m candles
LEADS = sorted({int(x) for x in os.getenv("ALERT_LEAD_MINUTES", "0,5").split(",")})
WEBHOOK = os.getenv("WEBHOOK_URL", "")  # Discord/Slack-style incoming webhook (optional)
NOTIFY_ON = {x.strip() for x in os.getenv("NOTIFY_ON", "BUY").split(",")}
NOTIFY_MODE = os.getenv("NOTIFY_MODE", "both")  # both = Claude AND quant agree; either = any one
TZ = ZoneInfo(os.getenv("TZ_NAME", "America/Los_Angeles"))
TFS = {"1m": 1, "5m": 5, "10m": 10, "15m": 15, "30m": 30, "1h": 60}
ROWS = {"1m": 15, "5m": 12, "10m": 10, "15m": 10, "30m": 8, "1h": 8}

ex = ccxt.coinbase({"enableRateLimit": True})
_client = None


def get_client():  # lazy: the app runs fine with no Anthropic key installed/configured
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic
        _client = AsyncAnthropic()
    return _client


_candles = pd.DataFrame()
state = {"latest": None, "history": deque(maxlen=100), "error": None, "next": None, "seq": 0, "threshold": None,
         "claude": bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv("USE_CLAUDE", "auto") != "0", "engine_err": None}
lock = asyncio.Lock()


# ---------------------------------------------------------------- market data
def _fetch_1m(since_ms):
    rows, now = [], ex.milliseconds()
    while since_ms < now:
        batch = ex.fetch_ohlcv(SYMBOL, "1m", since=since_ms, limit=300)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + 60_000
        if nxt <= since_ms:
            break
        since_ms = nxt
    return rows


def update_candles():
    """First call backfills LOOKBACK_MIN of 1m candles; later calls fetch only the new tail."""
    global _candles
    now = ex.milliseconds()
    since = now - LOOKBACK_MIN * 60_000 if _candles.empty else int(_candles.index[-1].value // 10**6)
    df = pd.DataFrame(_fetch_1m(since), columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    _candles = pd.concat([_candles, df]) if not _candles.empty else df
    _candles = _candles[~_candles.index.duplicated(keep="last")].sort_index().iloc[-(LOOKBACK_MIN + 60):]
    return _candles.copy()


def depth_summary():
    ob = ex.fetch_order_book(SYMBOL, limit=500)
    bids, asks = ob["bids"], ob["asks"]
    bb, ba = bids[0][0], asks[0][0]
    mid = (bb + ba) / 2
    bands = []
    for pct in (0.05, 0.1, 0.25, 0.5, 1.0):
        bv = sum(s for p, s in bids if p >= mid * (1 - pct / 100))
        av = sum(s for p, s in asks if p <= mid * (1 + pct / 100))
        full = bids[-1][0] <= mid * (1 - pct / 100) and asks[-1][0] >= mid * (1 + pct / 100)
        bands.append((pct, bv, av, (bv - av) / (bv + av) if bv + av else 0.0, full))
    wall = lambda side: sorted([x for x in side if abs(x[0] - mid) / mid <= 0.01], key=lambda x: -x[1])[:3]
    return {"bid": bb, "ask": ba, "mid": mid, "bands": bands, "bid_walls": wall(bids), "ask_walls": wall(asks),
            "cover": (bids[-1][0], asks[-1][0])}


# ---------------------------------------------------------------- indicators
def rma(s, n):  # Wilder smoothing
    return s.ewm(alpha=1 / n, adjust=False).mean()


def vol_stop(c, atr, mult=3.0):
    """ATR trailing stop ('Volatility Index/Stop', length 10, mult 3, Wilder smoothing)."""
    out, cur, up = [np.nan] * len(c), np.nan, True
    for i, (cv, av) in enumerate(zip(c.values, atr.values)):
        if np.isnan(av):
            continue
        if np.isnan(cur):
            cur = cv - mult * av
        if up:
            cur = max(cur, cv - mult * av)
            if cv < cur:
                up, cur = False, cv + mult * av
        else:
            cur = min(cur, cv + mult * av)
            if cv > cur:
                up, cur = True, cv - mult * av
        out[i] = cur
    return pd.Series(out, index=c.index)


def compute(df):
    c, h, l = df.close, df.high, df.low
    o = pd.DataFrame(index=df.index)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    o["bb_mid"], o["bb_up"], o["bb_lo"] = ma, ma + 2 * sd, ma - 2 * sd
    o["pctb"] = (c - o.bb_lo) / (o.bb_up - o.bb_lo)
    o["bw"] = (o.bb_up - o.bb_lo) / ma * 100
    d = c.diff()
    rsi = 100 - 100 / (1 + rma(d.clip(lower=0), 14) / rma((-d).clip(lower=0), 14).replace(0, np.nan))
    lo, hi = rsi.rolling(14).min(), rsi.rolling(14).max()
    k = ((rsi - lo) / (hi - lo).replace(0, np.nan) * 100).rolling(3).mean()
    o["k"], o["d"] = k, k.rolling(3).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    o["atr"] = rma(tr, 14)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    o["macd"], o["macd_sig"] = macd, macd.ewm(span=9, adjust=False).mean()
    o["macd_h"] = o.macd - o.macd_sig
    o["vstop"] = vol_stop(c, rma(tr, 10))
    return o


def resample(df, m):
    if m == 1:
        return df
    return df.resample(f"{m}min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def build_inds(df):
    """Indicator frames per timeframe, indexed by bar CLOSE time (no look-ahead)."""
    out = {}
    for name, m in TFS.items():
        r = resample(df, m)
        ind = compute(r).assign(close=r.close)
        ind.index = ind.index + pd.Timedelta(minutes=m)
        out[name] = ind
    return out


# ---------------------------------------------------------------- text report
def fmt(x, d=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}"


def build_report(df, depth, mode):
    now = pd.Timestamp.now(tz="UTC")
    floor = now.floor("min")
    L = [f"MARKET SNAPSHOT {SYMBOL} Coinbase | {now.astimezone(TZ):%Y-%m-%d %H:%M:%S %Z} | mode={mode}",
         "Params: BB(20,2) | StochRSI(14,14,3,3) | ATR(14,Wilder) | MACD(12,26,9) | VolStop(ATR10 x3 Wilder)",
         "Rows are oldest->newest, times local. '*' = still-forming candle. Volume in base units.",
         f"Last price: {fmt(df.close.iloc[-1])}", ""]
    snap = {}
    for name, m in TFS.items():
        r = resample(df, m)
        forming = bool(r.index[-1] + pd.Timedelta(minutes=m) > floor)
        if forming and mode == "closed":
            r, forming = r.iloc[:-1], False
        ind = compute(r)
        t, x = r.iloc[-1], ind.iloc[-1]
        L.append(f"=== {name} ({len(r)} bars) ===")
        L.append("time | O H L C | vol | BBup BBmid BBlo %B bw% | StK StD | ATR | MACDh | VolStop")
        for i in range(-min(ROWS[name], len(r)), 0):
            a, b = r.iloc[i], ind.iloc[i]
            star = "*" if forming and i == -1 else ""
            L.append(f"{r.index[i].astimezone(TZ):%m-%d %H:%M}{star} | {fmt(a.open)} {fmt(a.high)} {fmt(a.low)} {fmt(a.close)} | "
                     f"{fmt(a.volume, 3)} | {fmt(b.bb_up)} {fmt(b.bb_mid)} {fmt(b.bb_lo)} {fmt(b.pctb, 2)} {fmt(b.bw, 3)} | "
                     f"{fmt(b.k, 1)} {fmt(b.d, 1)} | {fmt(b.atr)} | {fmt(b.macd_h, 2)} | {fmt(b.vstop)}")
        bw_pct = ind.bw.dropna().rank(pct=True).iloc[-1] * 100 if ind.bw.notna().any() else float("nan")
        atr_ratio = x.atr / ind.atr.tail(50).mean() if ind.atr.notna().any() else float("nan")
        stoch = "overbought" if x.k > 80 else "oversold" if x.k < 20 else "neutral"
        L.append(f"Summary {name}: close vs BB = {fmt(x.pctb, 2)} %B | BW percentile {fmt(bw_pct, 0)} | Stoch {stoch}, "
                 f"K {'>' if x.k > x.d else '<'} D | ATR/50bar-avg {fmt(atr_ratio, 2)} | MACD hist {'+' if x.macd_h > 0 else '-'} "
                 f"({'rising' if x.macd_h > ind.macd_h.iloc[-2] else 'falling'}) | close {'above' if t.close > x.vstop else 'below'} VolStop")
        L.append("")
        snap[name] = {"close": t.close, "forming": forming, **{k: (None if pd.isna(v) else round(float(v), 4))
                      for k, v in x.items()}}
    L.append("=== ORDER BOOK DEPTH (Coinbase) ===")
    L.append(f"bid {fmt(depth['bid'])} ask {fmt(depth['ask'])} mid {fmt(depth['mid'])} spread {fmt(depth['ask'] - depth['bid'])}")
    L.append(f"book covers {fmt(depth['cover'][0])} .. {fmt(depth['cover'][1])}")
    for pct, bv, av, imb, full in depth["bands"]:
        L.append(f"within +/-{pct}%: bids {fmt(bv, 3)} asks {fmt(av, 3)} imbalance {fmt(imb, 2)}{'' if full else ' (book truncated)'}")
    L.append("largest bid walls (price,size): " + ", ".join(f"{fmt(p)}x{fmt(s, 3)}" for p, s in depth["bid_walls"]))
    L.append("largest ask walls (price,size): " + ", ".join(f"{fmt(p)}x{fmt(s, 3)}" for p, s in depth["ask_walls"]))
    return "\n".join(L), snap


# ---------------------------------------------------------------- Claude
NUM = {"type": ["number", "null"]}
PCT = {"type": ["integer", "null"], "minimum": 0, "maximum": 100}
TOOL = {
    "name": "trade_signal",
    "description": "Report the buy/wait/avoid decision for the next 15-minute window.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "WAIT", "AVOID"]},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "summary": {"type": "string"},
            "entry_low": NUM, "entry_high": NUM, "stop_loss": NUM, "target_1": NUM, "target_2": NUM,
            "p_above_threshold": PCT, "p_touch_threshold": PCT,
            "invalidation": {"type": "string", "description": "Specific level/condition that would change the view."},
            "bias_by_timeframe": {"type": "object", "properties": {
                k: {"type": "string", "enum": ["bullish", "bearish", "neutral"]} for k in TFS}},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "risks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "confidence", "summary", "bias_by_timeframe", "reasons", "risks", "invalidation"],
    },
}
SYSTEM = """You are a disciplined intraday crypto analyst. Input: a numeric multi-timeframe report (Bollinger Bands, Stochastic RSI, ATR, MACD, volume, ATR volatility stop), order-book depth, and a COMPUTED CONTEXT block with realized volatility and the expected 15-minute range.
Rules:
- Use only supplied numbers; never invent data. Say so if the last bar is still forming or data looks stale.
- The horizon is ~15 minutes, where moves are mostly noise. The base rate for up vs down is ~50/50. Move probabilities away from 50% only in proportion to evidence, and stay within 35-65% unless the threshold is more than ~1 sigma away (use the distance in the context block).
- 5m and 15m set direction; 1m is for timing only; 30m and 1h describe the regime (range vs trend). A squeeze (low BB-width percentile, falling ATR) means a breakout is likely but its direction is unknown; do not call direction from it alone.
- Overbought/oversold StochRSI is not a reversal signal on its own; it needs a K/D turn or price at a band. In a trend (price beyond VolStop, wide bands) extremes persist.
- Depth imbalance is weak evidence; a large wall within 0.25% of price matters more.
- BUY only if several timeframes agree, depth does not contradict, and a stop of 1-1.5x the 5m ATR gives at least 1.5:1 reward/risk. Otherwise WAIT (say what would trigger a BUY) or AVOID. Give price levels only for BUY.
- If a threshold question is asked, fill p_above_threshold (P(price above it at the +15m mark)) and p_touch_threshold (P(it trades through it during the window)).
- Always give an invalidation level/condition. Be concise. This is analysis, not financial advice."""


OFF_SIGNAL = {"action": "OFF", "confidence": 0, "summary": "Claude is off — quant engine only.", "bias_by_timeframe": {},
              "reasons": [], "risks": [], "invalidation": "-"}


async def ask_claude(report, question):
    msg = await get_client().messages.create(
        model=MODEL, max_tokens=1200, temperature=0.2, system=SYSTEM, tools=[TOOL],
        tool_choice={"type": "tool", "name": "trade_signal"},
        messages=[{"role": "user", "content": f"{report}\n\n{question}"}])
    return next(b.input for b in msg.content if b.type == "tool_use")


def should_notify(item):
    a, b = item["signal"]["action"], item["quant"]["action"]
    if NOTIFY_MODE == "either":
        return a in NOTIFY_ON or b in NOTIFY_ON
    return item["consensus"] in NOTIFY_ON


async def notify(item):
    if not WEBHOOK:
        return
    s, q = item["signal"], item["quant"]
    text = (f"{SYMBOL} {item['consensus']} @ {item['price']:,.2f} | Claude {s['action']} ({s['confidence']}%) | "
            f"Quant {q['action']} ({q['strength']})\n{s['summary']}")
    async with httpx.AsyncClient(timeout=10) as h:
        await h.post(WEBHOOK, json={"content": text, "text": text})


# ---------------------------------------------------------------- orchestration
async def run_analysis(mode="closed", trigger="manual", threshold=None):
    if threshold is not None:
        state["threshold"] = threshold or None
    async with lock:
        df = await asyncio.to_thread(update_candles)
        depth = await asyncio.to_thread(depth_summary)
        thr, px = state["threshold"], float(df.close.iloc[-1])
        report, _ = build_report(df, depth, mode)
        inds, sig = build_inds(df), quant.sigma_1m(df.close)
        q = quant.decide(inds, pd.Timestamp.now(tz="UTC").floor("min"), px, sig, depth["bands"][2][3], thr) or \
            {"action": "WAIT", "strength": 0, "composite": 0, "scores": {}, "reasons": ["insufficient history"]}
        report += "\n\n" + quant.context_text(px, sig, {n: float(v.atr.iloc[-1]) for n, v in inds.items()}, thr)
        question = (f"QUESTION: will price be above {thr:,.2f} at the +15m mark? Also give the BUY/WAIT/AVOID call."
                    if thr else "QUESTION: BUY, WAIT or AVOID for the next 15 minutes?")
        signal = await ask_claude(report, question) if state["claude"] else OFF_SIGNAL
        a, b = signal["action"], q["action"]
        state["seq"] += 1
        item = {"id": state["seq"], "time": datetime.now(TZ).isoformat(timespec="seconds"), "trigger": trigger,
                "mode": mode, "price": px, "threshold": thr, "signal": signal, "quant": q,
                "consensus": (a if a == b else "SPLIT") if state["claude"] else b, "snapshot": _snap(inds), "report": report}
        state["latest"] = item
        state["history"].appendleft({k: item[k] for k in ("id", "time", "trigger", "price", "signal", "consensus")}
                                    | {"quant": {"action": b, "strength": q["strength"]}})
        state["error"] = None
    if should_notify(item):
        try:
            await notify(item)
        except Exception as e:
            state["error"] = f"webhook: {e}"
    return item


def _snap(inds):
    return {n: {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in i.iloc[-1].items()} for n, i in inds.items()}


def next_fire(now):
    base = now.replace(second=0, microsecond=0)
    base -= timedelta(minutes=base.minute % 15)
    c = [(base + timedelta(minutes=15 * k) - timedelta(minutes=ld) + timedelta(seconds=3), ld)
         for k in range(0, 6) for ld in LEADS]
    return min(x for x in c if x[0] > now)


async def scheduler():
    while True:
        t, lead = next_fire(datetime.now(timezone.utc))
        state["next"] = {"at": t.astimezone(TZ).isoformat(timespec="seconds"), "lead": lead}
        await asyncio.sleep(max(0, (t - datetime.now(timezone.utc)).total_seconds()))
        try:
            await run_analysis("closed" if lead == 0 else "live", "candle close" if lead == 0 else f"T-{lead}m")
        except Exception as e:
            state["error"] = f"{type(e).__name__}: {e}"


@asynccontextmanager
async def lifespan(_):
    tasks = [asyncio.create_task(scheduler()), asyncio.create_task(engine_loop())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(title="Ticker Alerts", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"symbol": SYMBOL, "model": MODEL, "leads": LEADS, "next": state["next"], "error": state["error"],
            "threshold": state["threshold"], "claude": state["claude"], "latest_id": state["latest"]["id"] if state["latest"] else None}


@app.get("/api/latest")
def latest():
    return state["latest"] or {}


@app.get("/api/history")
def history():
    return list(state["history"])


@app.post("/api/run")
async def run_now(mode: str = "live", threshold: float | None = None):
    """threshold=<price> asks the 'over/under X at +15m' question; threshold=0 clears it."""
    try:
        return await run_analysis(mode if mode in ("live", "closed") else "live", "manual", threshold)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/report", response_class=PlainTextResponse)
async def report_only(mode: str = "live"):
    """The text Claude would see (without the LLM call)."""
    df = await asyncio.to_thread(update_candles)
    depth = await asyncio.to_thread(depth_summary)
    return build_report(df, depth, mode)[0]


@app.get("/api/backtest")
async def backtest_endpoint():
    df = await asyncio.to_thread(update_candles)
    return await asyncio.to_thread(lambda: quant.backtest(build_inds(df), df.close))


# ---------------------------------------------------------------- contract engine (primary; no LLM needed)
engine = contract.Engine()
_qc = {"key": None, "q": None}


async def post_webhook(text):
    if WEBHOOK:
        try:
            async with httpx.AsyncClient(timeout=10) as h:
                await h.post(WEBHOOK, json={"content": text, "text": text})
        except Exception as e:
            state["engine_err"] = f"webhook: {e}"


async def engine_loop():
    every, sent = float(os.getenv("TICK_SECONDS", "5")), 0
    while True:
        try:
            df = await asyncio.to_thread(update_candles)
            tk = await asyncio.to_thread(ex.fetch_ticker, SYMBOL)
            spot, ts = float(tk.get("last") or df.close.iloc[-1]), time.time()
            row = df.loc[df.index >= pd.Timestamp(ts - ts % 900, unit="s", tz="UTC")]
            strike = float(row.open.iloc[0]) if len(row) else spot  # window open price
            ind = compute(df.tail(400))
            x, p = ind.iloc[-1], ind.iloc[-2]
            key = df.index[-1]
            if _qc["key"] != key:  # re-run the multi-timeframe scorer once per new 1m candle
                _qc.update(key=key, q=quant.decide(build_inds(df), key, spot, quant.sigma_1m(df.close)))
            q = _qc["q"] or {"composite": 0.0, "strength": 0}
            if not any(pd.isna(v) for v in (x.k, p.k, x.pctb, x.atr)):
                engine.tick(ts, spot, strike, {"k": float(x.k), "k_prev": float(p.k), "pctb": float(x.pctb), "atr": float(x.atr),
                                               "atr_avg": float(ind.atr.tail(50).mean()), "lean": q["composite"], "strength": q["strength"]})
            for e in reversed([e for e in engine.log if e["id"] > sent]):
                if e["alert"]:
                    await post_webhook(f"{SYMBOL} {e['strategy']} {e['kind']}: {e['text']}")
            sent = engine.seq
            state["engine_err"] = None
        except Exception as e:
            state["engine_err"] = f"{type(e).__name__}: {e}"
        await asyncio.sleep(every)


@app.get("/api/engine")
def engine_view():
    return {**engine.snapshot(time.time()), "err": state["engine_err"], "claude": state["claude"]}


@app.post("/api/engine/config")
def engine_config(payload: dict):
    for k, v in payload.items():
        if k == "claude":
            state["claude"] = bool(v) and bool(os.getenv("ANTHROPIC_API_KEY"))
        elif k in engine.cfg:
            engine.cfg[k] = type(engine.cfg[k])(v)
    return {"cfg": engine.cfg, "claude": state["claude"]}


@app.post("/api/engine/reset")
def engine_reset():
    engine.reset()
    return {"ok": True}

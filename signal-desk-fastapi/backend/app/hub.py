"""Owns everything live: CCXT clients, one feed per (exchange, symbol), the engines, paper trading, SQLite writes and
the fan-out to WebSocket clients. Has no FastAPI import, so it can be driven (and tested) on its own."""
from __future__ import annotations

import asyncio
import re
import time
from typing import Awaitable, Callable

from . import aggregate, config, db, indicators, ml
from .constants import MIN_MS, STRAT_LABEL
from .engine import Engine, live_price
from .paper import Paper, apply as paper_apply, settle as paper_settle, summarize

now_ms = lambda: int(time.time() * 1000)  # noqa: E731
BASE_RE = re.compile(r"^[A-Z0-9]{2,12}$")
TF = "1m"


class ExchangeNotSupported(ValueError):
    pass


async def default_client_factory(ex_id: str):
    """ccxt.pro (WebSocket) when the exchange has it and CCXT_USE_WS=1, else plain async ccxt (REST polling)."""
    import ccxt
    if ex_id not in ccxt.exchanges:
        raise ExchangeNotSupported(f"'{ex_id}' is not a CCXT exchange id")
    mod = None
    if config.USE_WS:
        try:
            import ccxt.pro as ccxtpro
            if ex_id in ccxtpro.exchanges:
                mod = ccxtpro
        except Exception:
            mod = None
    if mod is None:
        import ccxt.async_support as mod  # type: ignore
    client = getattr(mod, ex_id)({"enableRateLimit": True})
    try:
        await client.load_markets()
    except BaseException:
        try:                      # a geo-blocked / down exchange is retried later; don't leak its HTTP session each time
            await client.close()
        except Exception:
            pass
        raise
    return client


class ExchangeConn:
    def __init__(self, ex_id: str, factory: Callable[[str], Awaitable]):
        self.id, self._factory, self._client, self._lock = ex_id, factory, None, asyncio.Lock()

    async def client(self):
        async with self._lock:
            if self._client is None:
                self._client = await self._factory(self.id)
            return self._client

    async def close(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass


class SymbolState:
    def __init__(self, exchange, base, quote):
        self.exchange, self.base, self.quote = exchange, base, quote
        self.symbol = f"{base}/{quote}"
        self.eng = Engine(MIN_MS)
        self.raw: dict[int, dict] = {}
        self.ticker: dict | None = None
        self.error: str | None = None
        self.ready = asyncio.Event()        # set once the first seed finished OR failed
        self.tasks: list[asyncio.Task] = []
        self.refs = 0
        self.idle_since = now_ms()
        self.last_touch = now_ms()
        self.always_on = False
        self.snap_minute = 0
        self.ml_p_up: float | None = None
        self.ml_busy = False


class AggState(SymbolState):
    """One base asset consolidated across several exchanges. Duck-types SymbolState (exchange = the agg id, quote = 'AGG')
    so _evaluate / _snapshot / card / paper / SQLite all work on it unchanged. Its candles are merged from the member
    SymbolStates' raw candles instead of arriving from a single exchange."""

    def __init__(self, aid: str, base: str):
        super().__init__(aid, base, "AGG")
        self.member_ids = aggregate.members(aid)
        self.feeds: dict[str, SymbolState] = {}     # exchange -> attached per-exchange feed
        self.venue_err: dict[str, str] = {}         # exchange -> why it is not contributing
        self.report: list[dict] = []
        self.written_for: frozenset = frozenset()   # which feeds the stored history was merged from
        self.ready.set()


class Session:
    """One WebSocket client."""

    def __init__(self, exchange, quote):
        self.exchange, self.quote = exchange, quote
        self.symbols: list[str] = []
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    def push(self, msg: dict):
        try:
            self.queue.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(msg)


class Hub:
    def __init__(self, client_factory=default_client_factory):
        self._factory = client_factory
        self.conns: dict[str, ExchangeConn] = {}
        self.syms: dict[tuple[str, str], SymbolState] = {}
        self.papers: dict[str, Paper] = {}
        self.sessions: set[Session] = set()
        self._bg: set[asyncio.Task] = set()
        self._tick_task: asyncio.Task | None = None

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        for item in config.ALWAYS_ON:
            ex, base, *q = item
            st = await self.ensure(ex, base.upper(), (q[0] if q else "USDT").upper())
            st.always_on = True
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self):
        if self._tick_task:
            self._tick_task.cancel()
        for st in list(self.syms.values()):
            for t in st.tasks:
                t.cancel()
        for t in list(self._bg):
            t.cancel()
        await asyncio.gather(*[c.close() for c in self.conns.values()], return_exceptions=True)

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return t

    def conn(self, ex_id) -> ExchangeConn:
        return self.conns.setdefault(ex_id, ExchangeConn(ex_id, self._factory))

    def paper(self, ex_id) -> Paper:
        if ex_id not in self.papers:
            self.papers[ex_id] = Paper(db.load_closed_trades(ex_id))
        return self.papers[ex_id]

    # ------------------------------------------------------------ symbols
    async def ensure(self, exchange, base, quote) -> SymbolState:
        key = (exchange, f"{base}/{quote}")
        st = self.syms.get(key)
        if st is None and aggregate.is_agg(exchange):
            st = self.syms[key] = AggState(exchange, base)
            st.tasks = [asyncio.create_task(self._run_agg(st))]
        elif st is None:
            st = self.syms[key] = SymbolState(exchange, base, quote)
            st.tasks = [asyncio.create_task(self._run_ohlcv(st)), asyncio.create_task(self._run_ticker(st))]
        st.last_touch = now_ms()
        return st

    async def _run_ohlcv(self, st: SymbolState):
        backoff = 1
        while True:
            try:
                client = await self.conn(st.exchange).client()
                if st.symbol not in client.markets:
                    st.error = f"{st.symbol} is not listed on {st.exchange}"
                    st.ready.set()
                    await asyncio.sleep(30)
                    continue
                if not st.raw:
                    self._ingest(st, await client.fetch_ohlcv(st.symbol, TF, limit=300), seed=True)
                    st.error = None
                    st.ready.set()
                ws = config.USE_WS and client.has.get("watchOHLCV") is True
                while True:
                    if ws:
                        rows = await client.watch_ohlcv(st.symbol, TF)
                    else:
                        await asyncio.sleep(config.POLL_S)
                        rows = await client.fetch_ohlcv(st.symbol, TF, limit=50)
                    self._ingest(st, rows)
                    st.error, backoff = None, 1
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network blips, rate limits, exchange maintenance: report, back off, retry
                st.error = f"{type(e).__name__}: {str(e)[:160]}"
                st.ready.set()
                await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)

    async def _run_ticker(self, st: SymbolState):
        backoff = 1
        while True:
            try:
                client = await self.conn(st.exchange).client()
                if st.symbol not in client.markets:
                    await asyncio.sleep(30)
                    continue
                ws = config.USE_WS and client.has.get("watchTicker") is True
                while True:
                    if ws:
                        t = await client.watch_ticker(st.symbol)
                    else:
                        await asyncio.sleep(5)
                        t = await client.fetch_ticker(st.symbol)
                    st.ticker = self._norm_ticker(t)
                    backoff = 1
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(min(backoff, 30))
                backoff = min(backoff * 2, 30)

    @staticmethod
    def _norm_ticker(t: dict) -> dict:
        return dict(ts=t.get("timestamp") or now_ms(), last=t.get("last") or t.get("close"), bid=t.get("bid"), ask=t.get("ask"),
                    high=t.get("high"), low=t.get("low"), base_volume=t.get("baseVolume"), quote_volume=t.get("quoteVolume"),
                    percentage=t.get("percentage"))

    # ------------------------------------------------------------ consolidated (multi-exchange) feeds
    async def _attach(self, agg: AggState, ex: str):
        """Resolve base/<USD-like quote> on one exchange and hold a reference to that exchange's normal feed."""
        client = await asyncio.wait_for(self.conn(ex).client(), 60)
        sym = aggregate.pick_symbol(client.markets, agg.base)
        if sym is None:
            raise LookupError(f"no {agg.base}/{'|'.join(config.AGG_QUOTES)} spot market")
        base, quote = sym.split("/")
        st = await self.ensure(ex, base, quote)
        st.refs += 1
        agg.feeds[ex] = st

    async def _run_agg(self, agg: AggState):
        """Attach every member exchange (in parallel; a slow or broken one never blocks the rest), retry the failures."""
        while True:
            todo = [ex for ex in agg.member_ids if ex not in agg.feeds]
            if not todo:
                return
            res = await asyncio.gather(*(self._attach(agg, ex) for ex in todo), return_exceptions=True)
            for ex, r in zip(todo, res):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                if isinstance(r, BaseException):
                    agg.venue_err[ex] = f"{type(r).__name__}: {str(r)[:120]}" if not isinstance(r, LookupError) else str(r)
                else:
                    agg.venue_err.pop(ex, None)
            if not agg.feeds and len(agg.venue_err) == len(agg.member_ids):
                agg.error = f"{agg.base}: no selected exchange has it (" + "; ".join(f"{k}: {v}" for k, v in list(agg.venue_err.items())[:3]) + ")"
            else:
                agg.error = None
            await asyncio.sleep(60 if agg.venue_err else 1)

    def _refresh_agg(self, agg: AggState, now: int):
        per = {ex: st.raw for ex, st in agg.feeds.items() if st.raw}
        if not per:
            return
        cs, stats = aggregate.merge_candles(per, limit=config.MAX_CANDLES)
        fresh = [st.eng.last_tick_at for st in agg.feeds.values() if st.eng.last_tick_at]
        used = frozenset(per)
        # first merge, or a venue joined: the whole consolidated history changed, so rewrite it (not just the last 3 candles)
        self._publish(agg, cs, now, seed=used != agg.written_for)
        agg.written_for = used
        agg.eng.last_tick_at = max(fresh) if fresh else 0   # staleness = freshest member, not "when we last merged"
        agg.ticker = aggregate.best_book({ex: st.ticker for ex, st in agg.feeds.items()}, now)
        if agg.ticker is not None:
            agg.ticker["last"] = live_price(agg.eng)
        price = live_price(agg.eng)
        rep = []
        for ex in agg.member_ids:
            st = agg.feeds.get(ex)
            if st is None:
                rep.append(dict(exchange=ex, status="error", symbol=None, price=None, devBps=None, error=agg.venue_err.get(ex, "connecting…")))
                continue
            p = live_price(st.eng)
            age = now - st.eng.last_tick_at if st.eng.last_tick_at else None
            status = "error" if st.error and not st.raw else "stale" if age is None or age > config.STALE_MS else "live"
            dev = (p - price) / price * 1e4 if p is not None and price else None
            rep.append(dict(exchange=ex, status=status, symbol=st.symbol, price=p, devBps=dev, error=st.error,
                            used=stats.get(ex, {}).get("used", 0), rejected=stats.get(ex, {}).get("rejected", 0)))
        agg.report = rep

    async def aggregate_snapshot(self, exchanges: list[str], base: str, limit=100, wait_s=15.0) -> dict:
        """REST helper: consolidated candles + latest indicators/signals for one asset across `exchanges`."""
        aid = aggregate.agg_id(exchanges)
        st = await self.ensure(aid, base, "AGG")
        st.last_touch = now_ms()
        deadline = time.time() + wait_s
        while time.time() < deadline:
            self._refresh_agg(st, now_ms())
            if st.eng.current is not None or st.error:
                break
            await asyncio.sleep(0.5)
        return dict(agg=aid, candles=[dict(c) for c in st.eng.candles[-limit:]], card=self.card(st, base, now_ms()), error=st.error)

    # ------------------------------------------------------------ data in
    def _ingest(self, st: SymbolState, rows, seed=False):
        for r in rows or []:
            if r[0] is None or r[4] is None:
                continue
            st.raw[int(r[0])] = dict(t=int(r[0]), o=float(r[1]), h=float(r[2]), l=float(r[3]), c=float(r[4]), v=float(r[5] or 0))
        if len(st.raw) > 600:
            for k in sorted(st.raw)[:-600]:
                del st.raw[k]
        self._publish(st, [st.raw[k] for k in sorted(st.raw)], now_ms(), seed)

    def _publish(self, st: SymbolState, cs: list[dict], now: int, seed=False):
        """Candles -> engine -> SQLite -> signals. Shared by single-exchange feeds and consolidated (agg) feeds."""
        closed = [k for k in cs if k["t"] + MIN_MS <= now]
        forming = next((k for k in cs if k["t"] + MIN_MS > now), None)
        changed = st.eng.update(closed, forming, now)
        if changed and closed:
            # (re)write the last two closed candles too: an exchange may still amend the candle that just closed
            ts = {k["t"] for k in (closed if seed else closed[-3:])}
            db.upsert_candles(st.exchange, st.symbol, TF, [k for k in closed if k["t"] in ts])
            db.upsert_indicators(st.exchange, st.symbol, TF, st.eng.frame, indicators.backend_name(), only_ts=ts)
            if config.ML_ENABLED:
                self._schedule_ml(st, closed[-1]["t"])
        self._evaluate(st, now)

    def _schedule_ml(self, st: SymbolState, ts: int):
        if st.ml_busy or len(st.eng.frame) < 40:
            return
        st.ml_busy = True
        frame = st.eng.frame.copy()

        async def job():
            try:
                p = await asyncio.to_thread(ml.PREDICTOR.p_up, frame)
                if p is not None:
                    st.ml_p_up = p
                    db.set_ml_p_up(st.exchange, st.symbol, TF, ts, p)
            except Exception as e:
                print(f"[ml] {st.symbol}: {e}")
            finally:
                st.ml_busy = False

        self._spawn(job())

    # ------------------------------------------------------------ signals + paper
    def _evaluate(self, st: SymbolState, now: int):
        eng = st.eng
        if eng.current is None:
            return
        sig = eng.evaluate(now)
        paper = self.paper(st.exchange)
        exits: list = []
        opened = paper_apply(paper, st.base, TF, MIN_MS, eng, sig, now, exits)
        price = live_price(eng)
        for o in opened:
            db.insert_signal(st.exchange, st.symbol, TF, o["stratKey"], o["signal"], ts=now, conf=o["conf"], ask=o.get("ask"),
                             lean=sig.get("lean"), story=sig.get("story"), price=price, strike=eng.contract_strike,
                             contract_start=eng.contract_start, reason=o.get("reason"))
        for x in exits:
            name = "STOP LEVEL" if x["kind"] == "stop" else "TAKE PROFIT"
            db.insert_signal(st.exchange, st.symbol, TF, x["stratKey"], name, ts=now, ask=x.get("bid"), story=sig.get("story"), price=price,
                             strike=eng.contract_strike, contract_start=eng.contract_start, reason=f"bid {x.get('bid')} vs level {x.get('level')}")
        self._drain(st.exchange, paper, now)
        if opened or exits:
            entries = [dict(ts=now, sym=st.base, tf=TF, strat=STRAT_LABEL[o["stratKey"]], signal=o["signal"], conf=o["conf"]) for o in opened]
            entries += [dict(ts=now, sym=st.base, tf=TF, strat=STRAT_LABEL[x["stratKey"]], signal="STOP LEVEL" if x["kind"] == "stop" else "TAKE PROFIT", conf=None) for x in exits]
            recs = [dict(kind="buy", sym=st.base, tf=TF, stratKey=o["stratKey"], signal=o["signal"], conf=o["conf"]) for o in opened]
            recs += [{**x, "tf": TF} for x in exits]
            for s in self.sessions:
                if s.exchange == st.exchange and st.base in s.symbols and s.quote == st.quote:
                    s.push(dict(type="log", entries=entries))
                    for r in recs:
                        s.push(dict(type="rec", rec=r))

    def _drain(self, exchange, paper: Paper, now):
        journal, paper.journal = paper.journal, []
        for event, rec in journal:
            symbol = f"{rec['sym']}/{self._quote_for(exchange, rec['sym'])}"
            db.save_trade(exchange, symbol, rec, "open" if event == "open" else "closed", now)

    def _quote_for(self, exchange, base):
        if aggregate.is_agg(exchange):
            return "AGG"
        for (ex, sym), st in self.syms.items():
            if ex == exchange and st.base == base:
                return st.quote
        return "USDT"

    # ------------------------------------------------------------ periodic work
    async def _tick_loop(self):
        while True:
            await asyncio.sleep(config.PUSH_S)
            try:
                self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # never let the loop die
                print(f"[tick] {type(e).__name__}: {e}")

    def tick(self, now: int | None = None):
        now = now or now_ms()
        by_ex: dict[str, dict] = {}
        for st in list(self.syms.values()):
            by_ex.setdefault(st.exchange, {})[(st.base, TF)] = st.eng
            if isinstance(st, AggState):
                self._refresh_agg(st, now)        # re-merge member candles (also re-evaluates)
            elif st.eng.current is not None:
                self._evaluate(st, now)           # time-driven changes (minutes remaining, settlement) need no new data
            minute = now // MIN_MS * MIN_MS
            if st.snap_minute != minute and st.eng.current is not None:
                st.snap_minute = minute
                self._snapshot(st, minute)
        for ex, engines in by_ex.items():
            paper = self.paper(ex)
            if paper_settle(paper, engines, now):
                self._drain(ex, paper, now)
        for s in list(self.sessions):
            s.push(self.state_message(s, now))
        self._reap(now)

    def _snapshot(self, st: SymbolState, minute: int):
        t = st.ticker if st.ticker and now_ms() - (st.ticker["ts"] or 0) < 90_000 else None
        if t:
            db.insert_ticker(st.exchange, st.symbol, minute, t, "ticker")
        else:  # exchange without a usable ticker feed: fall back to the last candle so the minute is never empty
            price = live_price(st.eng)
            db.insert_ticker(st.exchange, st.symbol, minute, dict(last=price), "candle")

    def _reap(self, now):
        for key, st in list(self.syms.items()):
            if st.refs > 0 or st.always_on:
                continue
            if now - max(st.idle_since, st.last_touch) > config.IDLE_STOP_S * 1000:
                for t in st.tasks:
                    t.cancel()
                if isinstance(st, AggState):
                    for m in st.feeds.values():
                        m.refs = max(0, m.refs - 1)
                        if m.refs == 0:
                            m.idle_since = now
                del self.syms[key]

    # ------------------------------------------------------------ sessions
    async def add_session(self, exchange, quote) -> Session:
        if not aggregate.is_agg(exchange):
            self.conn(exchange)
        s = Session(exchange, quote)
        self.sessions.add(s)
        return s

    async def set_symbols(self, s: Session, bases: list[str]):
        clean = []
        for b in bases:
            b = str(b).upper()
            if BASE_RE.match(b) and b not in clean:
                clean.append(b)
        clean = clean[:30]
        for b in clean:
            if b not in s.symbols:
                st = await self.ensure(s.exchange, b, s.quote)
                st.refs += 1
        for b in s.symbols:
            if b not in clean:
                self._release(s, b)
        s.symbols = clean

    def _release(self, s: Session, base):
        st = self.syms.get((s.exchange, f"{base}/{s.quote}"))
        if st:
            st.refs = max(0, st.refs - 1)
            if st.refs == 0:
                st.idle_since = now_ms()

    def remove_session(self, s: Session):
        for b in s.symbols:
            self._release(s, b)
        self.sessions.discard(s)

    def configure(self, s: Session, cfg: dict):
        self.paper(s.exchange).configure(fee_cents=cfg.get("feeCents"), mid_enabled=cfg.get("midEnabled"), exit_cfg=cfg.get("exitCfg"))

    # ------------------------------------------------------------ outbound
    def card(self, st: SymbolState | None, base: str, now: int) -> dict:
        if st is None:
            return dict(symbol=base, failed=False, error=None, price=None, strike=None, contractStart=None, ind=None, sig=None, spotBook=None, mlPUp=None)
        eng = st.eng
        tk = st.ticker if st.ticker and st.ticker.get("bid") is not None and st.ticker.get("ask") is not None else None
        return dict(symbol=base, failed=bool(st.error), error=st.error, price=live_price(eng), strike=eng.contract_strike,
                    contractStart=eng.contract_start, ind=eng.ind, sig=eng.signals, mlPUp=st.ml_p_up,
                    spotBook=dict(bid=tk["bid"], ask=tk["ask"]) if tk else None, lastTickAt=eng.last_tick_at,
                    venues=st.report if isinstance(st, AggState) else None)

    def state_message(self, s: Session, now: int) -> dict:
        sts = [self.syms.get((s.exchange, f"{b}/{s.quote}")) for b in s.symbols]
        live = [st for st in sts if st and st.eng.last_tick_at]
        if any(now - st.eng.last_tick_at <= config.STALE_MS for st in live):
            feed = "live"
        elif live or any(st and st.error for st in sts):
            feed = "stale"
        else:
            feed = "connecting"
        failed = [f"{s.exchange}: {st.base} failed — {st.error}" for st in sts if st and st.error]
        return dict(type="state", ts=now, exchange=s.exchange, feed=feed, note=failed[0] if failed else "",
                    indicatorBackend=indicators.backend_name(), aggMode=config.AGG_MODE if aggregate.is_agg(s.exchange) else None, cards=[self.card(st, b, now) for st, b in zip(sts, s.symbols)],
                    paper=summarize(self.paper(s.exchange), TF, list(s.symbols), now))

    # ------------------------------------------------------------ REST helpers
    async def candles(self, exchange, base, quote, limit=100, wait_s=10.0) -> tuple[SymbolState, list[dict]]:
        st = await self.ensure(exchange, base, quote)
        if not st.ready.is_set():
            try:
                await asyncio.wait_for(st.ready.wait(), wait_s)
            except asyncio.TimeoutError:
                pass
        cs = [st.raw[k] for k in sorted(st.raw)][-limit:]
        return st, cs

    async def ticker(self, exchange, base, quote, max_age_ms=90_000) -> dict:
        st = await self.ensure(exchange, base, quote)
        t = st.ticker
        if not t or now_ms() - (t["ts"] or 0) > max_age_ms:
            client = await self.conn(exchange).client()
            if st.symbol not in client.markets:
                raise KeyError(f"{st.symbol} is not listed on {exchange}")
            t = st.ticker = self._norm_ticker(await client.fetch_ticker(st.symbol))
        c = st.eng.current
        return dict(exchange=exchange, symbol=st.symbol, **t, candle=c, minute=now_ms() // MIN_MS * MIN_MS)

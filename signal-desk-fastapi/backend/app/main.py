"""Signal Desk backend.   Run from ./backend:   uvicorn app.main:app --port 8000

REST                                                    WebSocket
  GET  /api/health                                        /ws/desk?exchange=binance&quote=USDT   full desk (cards, paper, log, alerts)
                                                          /ws/desk?exchanges=binance,kraken,okx  same, but ONE consolidated feed across exchanges
  GET  /api/exchanges                                     /ws/tickers?exchange=binance&symbols=BTC,ETH   live ticker stream
  GET  /api/candles?exchange=&symbol=BTC&quote=USDT     (single exchange, raw candles)
  GET  /api/aggregate?exchanges=binance,kraken&symbol=BTC   consolidated candles + Bollinger/StochRSI/ATR/signals across exchanges
  GET  /api/tickers?exchange=&symbols=BTC,ETH           (fresh 1-minute ticker snapshots)
  GET  /api/db/{candles|indicators|ticker_snapshots|signals|paper_trades|ml_models}
  GET  /api/ml/status    POST /api/ml/train                      (h2o AutoML, as before)
  GET  /api/ml/approaches  POST /api/ml/compare  GET /api/ml/compare  POST /api/ml/approaches/train  GET /api/ml/predict   (multi-approach)
"""
from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import aggregate, config, db, indicators, ml
from .approaches import models as approach_models, service as approaches
from .engine import live_price
from .hub import BASE_RE, ExchangeNotSupported, Hub, now_ms

hub = Hub()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    await hub.start()
    yield
    await hub.stop()
    db.close()


app = FastAPI(title="Signal Desk backend", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- helpers
def clean(o):
    """json.dumps happily writes NaN/Infinity, which JSON.parse in the browser rejects. Turn them into null."""
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def ccxt_ids():
    try:
        import ccxt
        pro = []
        try:
            import ccxt.pro as ccxtpro
            pro = list(ccxtpro.exchanges)
        except Exception:
            pass
        return ccxt.__version__, list(ccxt.exchanges), pro
    except Exception:
        return None, [], []


def check_exchange(ex: str):
    ver, ids, _ = ccxt_ids()
    if ver is None:
        raise HTTPException(503, "ccxt is not installed on the backend (pip install ccxt)")
    if ex not in ids:
        raise HTTPException(400, f"'{ex}' is not a CCXT exchange id")


def parse_exchanges(csv: str) -> list[str]:
    ids = list(dict.fromkeys(x.strip().lower() for x in csv.split(",") if x.strip()))
    if not ids:
        raise HTTPException(400, "no exchanges given")
    if len(ids) > config.MAX_AGG_EXCHANGES:
        raise HTTPException(400, f"at most {config.MAX_AGG_EXCHANGES} exchanges at once (MAX_AGG_EXCHANGES)")
    for ex in ids:
        check_exchange(ex)
    return ids


def split_symbol(symbol: str, quote: str) -> tuple[str, str]:
    base = symbol
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
    base, quote = base.upper(), quote.upper()
    if not BASE_RE.match(base) or not BASE_RE.match(quote):
        raise HTTPException(400, f"bad symbol '{symbol}'")
    return base, quote


# ---------------------------------------------------------------- REST
@app.get("/api/health")
async def health():
    ver, ids, pro = ccxt_ids()
    return dict(ok=True, ccxt=ver, ccxt_exchanges=len(ids), ccxt_pro_exchanges=len(pro), websocket_feeds=config.USE_WS,
                indicator_backend=indicators.backend_name(), atr_smoothing=config.ATR_SMOOTHING, db=config.DB_PATH,
                feeds=[f"{ex}:{sym}" for ex, sym in hub.syms], clients=len(hub.sessions), ml=ml.status())


@app.get("/api/exchanges")
async def exchanges():
    ver, ids, pro = ccxt_ids()
    return dict(ccxt=ver, exchanges=ids, pro=pro)


@app.get("/api/candles")
async def candles(exchange: str, symbol: str, quote: str = "USDT", limit: int = Query(100, ge=1, le=600)):
    """Normalized [{t,o,h,l,c,v}] (oldest first, last one is still forming) from the live CCXT feed."""
    check_exchange(exchange)
    base, quote = split_symbol(symbol, quote)
    st, cs = await hub.candles(exchange, base, quote, limit)
    if not cs:
        if st.error and "not listed" in st.error:
            raise HTTPException(404, st.error)
        if st.error:
            raise HTTPException(502, st.error)
        raise HTTPException(503, "feed is still starting")
    return dict(exchange=exchange, symbol=st.symbol, candles=cs, error=st.error)


@app.get("/api/aggregate")
async def aggregate_view(exchanges: str, symbol: str, limit: int = Query(100, ge=1, le=400), wait_s: float = Query(15, ge=1, le=60)):
    """The FastAPI algorithm (ta Bollinger / StochRSI / ATR + the three strategies) applied to candles merged across CCXT exchanges."""
    ids = parse_exchanges(exchanges)
    base = symbol.split("/")[0].upper()
    if not BASE_RE.match(base):
        raise HTTPException(400, f"bad symbol '{symbol}'")
    out = await hub.aggregate_snapshot(ids, base, limit, wait_s)
    return clean(out)


@app.get("/api/tickers")
async def tickers(exchange: str, symbols: str, quote: str = "USDT"):
    check_exchange(exchange)
    out = []
    for s in [x for x in symbols.split(",") if x][:30]:
        base, q = split_symbol(s, quote)
        try:
            out.append(clean(await hub.ticker(exchange, base, q)))
        except Exception as e:
            out.append(dict(exchange=exchange, symbol=f"{base}/{q}", error=f"{type(e).__name__}: {str(e)[:160]}"))
    return dict(ts=now_ms(), tickers=out)


@app.get("/api/db/{table}")
async def read_table(table: str, exchange: str | None = None, symbol: str | None = None, tf: str | None = None,
                     since: int | None = None, limit: int = Query(500, ge=1, le=5000), order: str = "desc"):
    """symbol may be BTC/USDT or just BTC (uses quote=USDT)."""
    if table not in db.TABLES:
        raise HTTPException(404, f"tables: {sorted(db.TABLES)}")
    if symbol and "/" not in symbol:
        symbol = f"{symbol.upper()}/USDT"
    return clean(dict(table=table, rows=db.query(table, exchange=exchange, symbol=symbol, tf=tf, since=since, limit=limit, order=order)))


class TrainRequest(BaseModel):
    exchange: str
    symbols: list[str] = Field(min_length=1)          # unified, e.g. ["BTC/USDT","ETH/USDT"]
    tf: str = "1m"
    horizon: int | None = None
    max_runtime_secs: int = Field(120, ge=10, le=3600)
    max_models: int = Field(8, ge=1, le=50)


@app.get("/api/ml/status")
async def ml_status():
    return clean(ml.status())


@app.post("/api/ml/train", status_code=202)
def ml_train(req: TrainRequest, background: BackgroundTasks):
    if not ml._has_h2o():
        raise HTTPException(501, "h2o is not installed (pip install h2o; it also needs a Java runtime)")
    if ml.STATUS.get("state") in ("building dataset", "training"):
        raise HTTPException(409, "a training run is already in progress")
    background.add_task(_train_quietly, req)
    return dict(started=True, poll="/api/ml/status")


def _train_quietly(req: TrainRequest):
    try:
        ml.train(req.exchange, req.symbols, req.tf, req.horizon, req.max_runtime_secs, req.max_models)
    except Exception as e:  # ml.train already recorded it in ml.STATUS
        print(f"[ml] training failed: {e}")


# ---------------------------------------------------------------- multi-approach ML (see app/approaches)
class CompareRequest(BaseModel):
    exchange: str
    symbols: list[str] = Field(min_length=1)
    tf: str = "1m"
    approaches: list[str] | None = None            # default: every available one + baselines
    n_folds: int = Field(4, ge=2, le=8)
    horizon: int | None = Field(None, ge=1, le=120)
    cost_bps: float | None = Field(None, ge=0, le=100)   # all-in round-trip friction the label and simulation must clear
    k: float | None = Field(None, ge=0, le=3)            # label threshold = cost + k * sigma * sqrt(H)
    thr: float | None = Field(None, gt=0.5, lt=1)        # trade if p >= thr (long) or p <= 1 - thr (short)
    gate_mult: float | None = Field(None, ge=0, le=10)   # volatility gate: expected move must be >= gate_mult * cost


class ApproachTrainRequest(CompareRequest):
    approach: str = "lightgbm"


def _busy():
    return approaches.COMPARE.get("state") in ("building dataset", "comparing") or approaches.TRAIN.get("state") == "training"


@app.get("/api/ml/approaches")
async def ml_approaches():
    return clean(dict(catalog=approach_models.catalog(), defaults=approaches.DEFAULTS, saved=approaches.saved_models(),
                      train=approaches.TRAIN, compare_state=approaches.COMPARE.get("state")))


@app.post("/api/ml/compare", status_code=202)
def ml_compare(req: CompareRequest, background: BackgroundTasks):
    unknown = [a for a in (req.approaches or []) if a not in approach_models.REGISTRY]
    if unknown:
        raise HTTPException(422, f"unknown approach(es) {unknown}; choose from {sorted(approach_models.REGISTRY)}")
    if _busy():
        raise HTTPException(409, "another approaches job is running")
    kw = req.model_dump(exclude={"exchange", "symbols", "tf", "approaches", "n_folds"})
    background.add_task(_quietly, approaches.run_compare, req.exchange, req.symbols, req.tf, req.approaches, req.n_folds, **kw)
    return dict(started=True, poll="/api/ml/compare")


@app.get("/api/ml/compare")
async def ml_compare_result():
    return clean(approaches.COMPARE)


@app.post("/api/ml/approaches/train", status_code=202)
def ml_approach_train(req: ApproachTrainRequest, background: BackgroundTasks):
    cls = approach_models.REGISTRY.get(req.approach)
    if cls is None or not cls.fitted:
        raise HTTPException(422, f"{req.approach!r} is not a trainable approach")
    if not cls.available():
        raise HTTPException(501, f"{req.approach} needs `pip install {cls.needs}`")
    if _busy():
        raise HTTPException(409, "another approaches job is running")
    kw = req.model_dump(exclude={"exchange", "symbols", "tf", "approaches", "n_folds", "approach"})
    background.add_task(_quietly, approaches.train_final, req.approach, req.exchange, req.symbols, req.tf, **kw)
    return dict(started=True, poll="/api/ml/approaches")


@app.get("/api/ml/predict")
async def ml_predict(exchange: str, symbol: str, tf: str = "1m", approach: str | None = None):
    try:
        return clean(await asyncio.to_thread(approaches.predict, exchange, symbol, tf, approach))
    except LookupError as e:
        raise HTTPException(404, str(e))


def _quietly(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:  # the job already recorded the error in its STATUS dict
        print(f"[approaches] {fn.__name__} failed: {e}")


# ---------------------------------------------------------------- WebSocket
@app.websocket("/ws/desk")
async def ws_desk(ws: WebSocket, exchange: str | None = None, exchanges: str | None = None, quote: str = "USDT"):
    await ws.accept()
    try:
        if exchanges:                      # consolidated: every listed exchange feeds ONE candle series per asset
            exchange, quote = aggregate.agg_id(parse_exchanges(exchanges)), "AGG"
        elif exchange:
            check_exchange(exchange)
        else:
            raise HTTPException(400, "give exchange=<id> or exchanges=<id,id,...>")
    except HTTPException as e:
        await ws.send_json(dict(type="error", detail=e.detail))
        await ws.close(code=1008)
        return
    session = await hub.add_session(exchange, quote.upper())

    async def sender():
        while True:
            await ws.send_json(clean(await session.queue.get()))

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await ws.receive_json()
            op = msg.get("op")
            if op == "subscribe":
                await hub.set_symbols(session, msg.get("symbols", []))
            elif op == "config":
                hub.configure(session, msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        send_task.cancel()
        hub.remove_session(session)


@app.websocket("/ws/tickers")
async def ws_tickers(ws: WebSocket, exchange: str, symbols: str, quote: str = "USDT"):
    await ws.accept()
    try:
        check_exchange(exchange)
        pairs = [split_symbol(s, quote) for s in symbols.split(",") if s][:30]
    except HTTPException as e:
        await ws.send_json(dict(type="error", detail=e.detail))
        await ws.close(code=1008)
        return
    try:
        while True:
            out = []
            for base, q in pairs:
                st = await hub.ensure(exchange, base, q)   # also keeps the feed alive while this socket is open
                out.append(dict(symbol=st.symbol, price=live_price(st.eng), error=st.error, **(st.ticker or {})))
            await ws.send_json(clean(dict(type="tickers", ts=now_ms(), exchange=exchange, tickers=out)))
            await asyncio.sleep(config.PUSH_S)
    except (WebSocketDisconnect, RuntimeError):
        pass

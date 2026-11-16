"""FastAPI router. Mounted from main.py:   from .quant15.routes import router as quant15_router ; app.include_router(quant15_router)"""
from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from . import bars, cdd, live, model15, service

router = APIRouter(prefix="/api/quant15", tags=["quant15"])


def _clean(o):  # json.dumps writes NaN; browsers reject it
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _quietly(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except Exception as e:  # the job recorded the error in its state dict
        print(f"[quant15] {fn.__name__} failed: {e}")


def _busy():
    return service.COMPARE.get("state") not in ("idle", "done", "error") or service.TRAIN.get("state") == "training" \
        or BACKFILL.get("state") == "running"


BACKFILL: dict = {"state": "idle"}


class BackfillRequest(BaseModel):
    exchange: str                                  # one real exchange id (binance, okx, ...), not an agg:... consolidation
    symbols: list[str] = Field(min_length=1)       # ["BTC/USDT","ETH/USDT"]
    days: int = Field(30, ge=1, le=1500)
    tf: str = Field("1m", pattern="^(1m|15m)$")    # 1m: what the replay needs; 15m: cheap long history for the model
    source: str = Field("auto", pattern="^(auto|cdd|ccxt)$")   # auto: CryptoDataDownload for binance 1m (then a CCXT top-up), CCXT otherwise


class CompareRequest(BaseModel):
    exchange: str
    symbols: list[str] = Field(min_length=1)
    approaches: list[str] | None = None            # default: lightgbm, catboost, xgboost (whichever are installed; else hist_gb)
    auto_fetch: bool = True                        # pull missing 1m history first: CryptoDataDownload for binance, CCXT for everything else
    days: int = Field(45, ge=1, le=1500)           # how much 1m history the CCXT top-up fetches when the database is short
    n_folds: int = Field(4, ge=2, le=8)
    margin: float = Field(0.02, ge=0, lt=0.5)      # a model "agrees" only if |p_up - 0.5| >= margin in the signal's direction
    vol_method: str = Field("har", pattern="^(har|ewma|garch)$")


class TrainRequest(BaseModel):
    exchange: str
    symbols: list[str] = Field(min_length=1)
    approaches: list[str] | None = None
    auto_fetch: bool = True
    days: int = Field(45, ge=1, le=1500)


class PolicyRequest(BaseModel):
    policy: str = Field(pattern="^(original|gated)$")
    margin: float | None = Field(None, ge=0, lt=0.5)


def _backfill_one(req: BackfillRequest, s: str) -> dict:
    use_cdd = req.tf == "1m" and req.exchange in cdd.SUPPORTED and req.source in ("auto", "cdd")
    if not use_cdd:
        return bars.backfill(req.exchange, s, req.days, req.tf)
    try:
        out = cdd.fetch_into(s, req.exchange, req.days)
    except Exception as e:  # noqa: BLE001
        if req.source == "cdd":
            raise
        return dict(bars.backfill(req.exchange, s, req.days, req.tf), note=f"CryptoDataDownload failed ({e}); used CCXT")
    if req.source == "auto":       # the CDD file ends about a day ago: let CCXT fill the recent stretch if it can reach the exchange
        try:
            out["topup"] = bars.backfill(req.exchange, s, req.days, req.tf)
        except Exception as e:  # noqa: BLE001
            out["topup"] = f"skipped: {type(e).__name__}: {e}"
    return out


def _run_backfill(req: BackfillRequest):
    BACKFILL.clear(); BACKFILL.update(state="running", results=[])
    try:
        for s in req.symbols:
            BACKFILL["results"].append(_backfill_one(req, s))
        BACKFILL["state"] = "done"
    except Exception as e:
        BACKFILL.update(state="error", error=str(e))
        raise


@router.post("/backfill", status_code=202)
def backfill(req: BackfillRequest, background: BackgroundTasks):
    if _busy():
        raise HTTPException(409, "a quant15 job is running")
    background.add_task(_quietly, _run_backfill, req)
    return dict(started=True, poll="/api/quant15/status")


def _wants_fetch(exchange: str, auto: bool) -> bool:
    return auto


def _ensure_one(exchange: str, s: str, days: int) -> dict:
    if exchange in cdd.SUPPORTED:
        return cdd.ensure(s, exchange)
    have = cdd.windows(exchange, s)
    if have >= cdd.MIN_WINDOWS:
        return dict(symbol=s, exchange=exchange, source="db", fetched=False, windows=have)
    out = bars.backfill(exchange, s, days, "1m")
    res = dict(out, source="ccxt", windows=cdd.windows(exchange, s))
    if out.get("fetched", 0) < 0.5 * days * 1440 and res["windows"] < cdd.MIN_WINDOWS:
        res["warning"] = (f"only {out.get('fetched', 0)} one-minute candles came back for {days} days: {s} on {exchange} is thinly traded or has "
                          "short public history. Try another quote (BTC/USD, BTC/USDC) or exchange id (coinbaseexchange).")
    return res


def _ensure_then(fn, exchange: str, symbols: list[str], auto: bool, days: int, *args):
    """Top up missing 1m history (CryptoDataDownload for binance, CCXT for any other exchange), then run the job. A failed
    fetch is recorded in /status and the job still runs, so it reports its own precise 'need N bars' error if the data really is too short."""
    if auto:
        for s in dict.fromkeys(list(symbols) + [model15.peer_symbol(x) for x in symbols]):
            try:
                BACKFILL["results"].append(_ensure_one(exchange, s, days))
            except Exception as e:  # noqa: BLE001  one bad symbol must not hide the others' results
                BACKFILL["results"].append(dict(symbol=s, exchange=exchange, error=f"{type(e).__name__}: {e}"))
        errs = [r for r in BACKFILL["results"] if r.get("error")]
        BACKFILL.update(state="error" if errs else "done", **({"error": "; ".join(f"{r['symbol']}: {r['error']}" for r in errs)} if errs else {}))
    try:
        fn(exchange, symbols, *args)
    except Exception as e:  # noqa: BLE001  put the fetch outcome next to the job error, so one /status call explains a short-history failure
        state = service.TRAIN if fn is service.train_final else service.COMPARE
        rows = [f"{r.get('symbol')}: " + (r["error"] if r.get("error") else f"{r.get('source')} fetched={r.get('fetched')} windows={r.get('windows')}")
                for r in BACKFILL.get("results", [])]
        if auto and state.get("state") == "error":
            state["error"] = f"{state.get('error', e)} | history fetch: {'; '.join(rows) or 'nothing ran'}"
        raise


@router.post("/compare", status_code=202)
def compare(req: CompareRequest, background: BackgroundTasks):
    if _busy():
        raise HTTPException(409, "a quant15 job is running")
    auto = _wants_fetch(req.exchange, req.auto_fetch)
    if auto:
        BACKFILL.clear(); BACKFILL.update(state="running", results=[])
    background.add_task(_quietly, _ensure_then, service.run_compare, req.exchange, req.symbols, auto, req.days, req.approaches, req.n_folds, req.margin, req.vol_method)
    return dict(started=True, poll="/api/quant15/status")


@router.post("/train", status_code=202)
def train(req: TrainRequest, background: BackgroundTasks):
    if _busy():
        raise HTTPException(409, "a quant15 job is running")
    auto = _wants_fetch(req.exchange, req.auto_fetch)
    if auto:
        BACKFILL.clear(); BACKFILL.update(state="running", results=[])
    background.add_task(_quietly, _ensure_then, service.train_final, req.exchange, req.symbols, auto, req.days, req.approaches)
    return dict(started=True, poll="/api/quant15/status")


@router.get("/policy")
async def get_policy():
    return _clean(live.status())


@router.post("/policy")
async def set_policy(req: PolicyRequest):
    live.set_policy(req.policy, req.margin)
    return _clean(live.status())


@router.get("/data")
async def data(exchange: str | None = None):
    return _clean(await asyncio.to_thread(service.inventory, exchange))


@router.get("/status")
async def status():
    return _clean(dict(backfill=BACKFILL, **service.status()))


@router.get("/decide")
async def decide(exchange: str, symbol: str, policy: str | None = Query(None, pattern="^(original|gated|model_only)$"),
                 margin: float | None = Query(None, ge=0, lt=0.5)):
    try:
        return _clean(await asyncio.to_thread(service.decide, exchange, symbol, policy, margin))
    except LookupError as e:
        raise HTTPException(404, str(e))

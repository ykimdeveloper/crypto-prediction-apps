"""FastAPI router. Mounted from main.py with two lines. With the toggle off nothing is computed and no candles are read."""
from __future__ import annotations

import asyncio
import math

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from . import backfill, config, forward_log, scoring, service, training

router = APIRouter(prefix="/api/levels", tags=["levels"])


def _clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class ConfigPatch(BaseModel):
    enabled: bool | None = None
    tfs: list[str] | None = None
    spans: list[str] | None = None
    windows: int | None = Field(None, ge=1)
    min_touches: int | None = Field(None, ge=0)
    confluence_only: bool | None = None
    show: str | None = None
    horizon_min: float | None = Field(None, gt=0)
    x_atr: float | None = Field(None, gt=0)
    zone_width_atr: float | None = Field(None, gt=0)
    round_numbers: bool | None = None
    journal: bool | None = None


@router.get("/config")
def get_config():
    return config.get()


@router.post("/config")
def set_config(p: ConfigPatch):
    try:
        return config.update(p.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(400, str(e))


def _csv(s):
    return [x for x in (s or "").split(",") if x] or None


@router.get("")
async def levels(exchange: str, symbol: str, tfs: str | None = None, spans: str | None = None, min_touches: int | None = Query(None, ge=0),
                 windows: int | None = Query(None, ge=1), as_of: int | None = None):
    cfg = config.get()
    if not cfg["enabled"]:
        return dict(enabled=False)
    if min_touches is not None:
        cfg["min_touches"] = min_touches
    try:
        return _clean(await asyncio.to_thread(service.levels_for, exchange, symbol, cfg, _csv(tfs), _csv(spans), as_of, None, windows))
    except KeyError as e:
        raise HTTPException(400, f"unknown timeframe or span: {e}")


@router.get("/zone/{zid}")
async def zone(zid: str, exchange: str, symbol: str, offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=1000), windows: int | None = Query(None, ge=1), as_of: int | None = None):
    cfg = config.get()
    if not cfg["enabled"]:
        return dict(enabled=False)
    try:
        r = await asyncio.to_thread(service.zone_detail, exchange, symbol, zid, cfg, as_of, None, windows)
    except (KeyError, ValueError):
        raise HTTPException(400, "zone id looks like 15m:1w:71234.50")
    z = r["zone"]
    if z is None:
        raise HTTPException(404, "zone not found for this as-of time (levels are recomputed from closed candles)")
    n = len(z["touches"])
    z = {**z, "touches": z["touches"][offset:offset + limit], "touches_total": n, "offset": offset}
    return _clean(dict(enabled=True, cell=r["cell"], zone=z))


@router.get("/journal")
async def journal(exchange: str, symbol: str, limit: int = Query(50, ge=1, le=500)):
    if not config.get()["enabled"]:
        return dict(enabled=False)
    return _clean(dict(enabled=True, windows=await asyncio.to_thread(forward_log.recent, forward_log._shared(), exchange, symbol, limit)))


@router.get("/journal/summary")
async def journal_summary(exchange: str, symbol: str):
    if not config.get()["enabled"]:
        return dict(enabled=False)
    return _clean(dict(enabled=True, **await asyncio.to_thread(forward_log.summary, forward_log._shared(), exchange, symbol)))


class TrainRequest(BaseModel):
    exchange: str
    symbols: list[str] = Field(min_length=1)
    n_checkpoints: int = Field(40, ge=10, le=200)
    step_hours: float = Field(6.0, gt=0, le=72)
    holdout_frac: float = Field(0.2, ge=0.1, le=0.5)
    approach: str | None = None
    tfs: list[str] | None = None
    spans: list[str] | None = None


def _run_train(req: TrainRequest):
    try:
        training.run_train(req.exchange, req.symbols, req.n_checkpoints, req.step_hours, req.holdout_frac, req.approach, req.tfs, req.spans)
    except Exception as e:  # noqa: BLE001  the failure is already in /train/status
        print(f"[levels] training failed: {type(e).__name__}: {e}")


@router.post("/train", status_code=202)
def train(req: TrainRequest, background: BackgroundTasks):
    if not config.get()["enabled"]:
        raise HTTPException(409, "Levels is Off; turn it on before training")
    if training.TRAIN.get("state") not in (None, "idle", "done", "done_no_model", "error"):
        raise HTTPException(409, "a levels training job is already running")
    training.TRAIN.update(state="queued")
    background.add_task(_run_train, req)
    return dict(started=True, poll="/api/levels/train/status")


@router.get("/train/status")
def train_status():
    return training.status()


@router.get("/grid")
async def grid(exchange: str, symbol: str, zones: bool = False, as_of: int | None = None):
    cfg = config.get()
    if not cfg["enabled"]:
        return dict(enabled=False)
    return _clean(dict(enabled=True, **await asyncio.to_thread(service.grid, exchange, symbol, cfg, as_of, None, zones)))


@router.get("/report")
async def report(exchange: str, symbol: str | None = None):
    if not config.get()["enabled"]:
        return dict(enabled=False)
    rep = scoring.load_report(exchange)
    out = dict(enabled=True, exchange=exchange, training_report=rep or dict(status="not_trained", note="POST /api/levels/train first"))
    if symbol:
        out["forward_journal"] = await asyncio.to_thread(forward_log.summary, forward_log._shared(), exchange, symbol)
    return _clean(out)


class BackfillRequest(BaseModel):
    exchange: str                                   # one real exchange id (coinbase, binance, ...), not an agg:... consolidation
    symbols: list[str] = Field(min_length=1)
    tfs: list[str] = Field(default_factory=lambda: list(backfill.NATIVE_TFS))
    days: int = Field(730, ge=1, le=800)


def _run_backfill(req: BackfillRequest):
    try:
        backfill.run(req.exchange, req.symbols, req.tfs, req.days)
    except Exception as e:  # noqa: BLE001  already recorded in /backfill/status
        backfill.STATE.update(state="error", error=f"{type(e).__name__}: {e}")
        print(f"[levels] backfill failed: {type(e).__name__}: {e}")


@router.post("/backfill", status_code=202)
def start_backfill(req: BackfillRequest, background: BackgroundTasks):
    if not config.get()["enabled"]:
        raise HTTPException(409, "Levels is Off; turn it on before backfilling")
    bad = [t for t in req.tfs if t not in service.TF_MS]
    if bad:
        raise HTTPException(400, f"unknown timeframes {bad}; choose from {list(service.TF_MS)}")
    if backfill.STATE.get("state") == "running":
        raise HTTPException(409, "a levels backfill is already running")
    backfill.STATE.update(state="running", results=[])
    background.add_task(_run_backfill, req)
    return dict(started=True, poll="/api/levels/backfill/status")


@router.get("/backfill/status")
def backfill_status():
    return backfill.status()


@router.get("/coverage")
async def coverage(exchange: str, symbol: str):
    if not config.get()["enabled"]:
        return dict(enabled=False)
    return _clean(dict(enabled=True, **await asyncio.to_thread(backfill.coverage, exchange, symbol)))

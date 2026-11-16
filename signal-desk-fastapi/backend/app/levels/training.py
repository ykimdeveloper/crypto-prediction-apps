"""Background training job (one at a time, polled like quant15). Blocking: the route runs it in a BackgroundTask."""
from __future__ import annotations

import json
import threading
import time

import pandas as pd
from joblib import dump

from .. import config as app_config
from . import config as levels_config, ml, scoring
from .data import SPAN_MS, TF_MS, load

_lock = threading.Lock()
TRAIN: dict = dict(state="idle")


def run_train(exchange, symbols, n_cp=40, step_hours=6.0, holdout_frac=0.2, approach=None, tfs=None, spans=None, loader=None):
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another levels training job is running")
    loader = loader or load
    try:
        cfg = levels_config.get()
        tfs, spans = tfs or cfg["tfs"], spans or cfg["spans"]
        TRAIN.clear()
        TRAIN.update(state="building events", started=int(time.time()), exchange=exchange, symbols=symbols)
        parts = []
        for s in symbols:
            frames = {tf: loader(exchange, s, tf) for tf in tfs}
            if any(f.empty for f in frames.values()):
                raise ValueError(f"no stored candles for {s} on {[t for t, f in frames.items() if f.empty]}; backfill first (see the data policy: 5m/15m/30m/1h natively)")
            TRAIN["state"] = f"building events for {s}"
            parts.append(ml.build_events(frames, s, cfg, tfs, spans, n_cp, int(step_hours * 3_600_000)))
        ev = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
        if ev.empty:
            raise ValueError("no resolved touch events; the candle history is too short for the selected cells")
        TRAIN["state"] = "validating"
        bundle, report = ml.train_report(ev, holdout_frac, approach=approach)
        report = dict(report, exchange=exchange, symbols=symbols, tfs=tfs, spans=spans, checkpoints=n_cp, step_hours=step_hours)
        app_config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if bundle is not None:
            dump(dict(bundle, exchange=exchange, symbols=symbols), scoring.bundle_path(exchange))
            scoring.forget(exchange)
        scoring.report_path(exchange).write_text(json.dumps(report))
        TRAIN.update(state="done" if bundle is not None else "done_no_model", finished=int(time.time()), result=report.get("status"), verdicts=report.get("verdicts"))
        return report
    except Exception as e:
        TRAIN.update(state="error", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        _lock.release()


def status():
    return dict(TRAIN)

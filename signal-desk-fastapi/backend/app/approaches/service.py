"""Entry points the FastAPI routes call. Blocking - run them in a worker thread (the routes use BackgroundTasks / to_thread)."""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd
from joblib import dump, load

from .. import config, db
from ..constants import MIN_MS
from . import evaluate, models, vol
from .features import FEATURES, build_features, to_frame

DEFAULTS = dict(horizon=15, cost_bps=8.0, k=0.5, thr=0.55, gate_mult=1.5)
COMPARE: dict = {"state": "idle"}
TRAIN: dict = {"state": "idle"}
_lock = threading.Lock()


def _registry_path():
    return config.MODEL_DIR / "approaches.json"


def _registry() -> list[dict]:
    p = _registry_path()
    return json.loads(p.read_text()) if p.exists() else []


def load_frames(exchange: str, symbols: list[str], tf: str = "1m") -> dict[str, pd.DataFrame]:
    return {s: to_frame(db.dataset(exchange, s, tf)) for s in symbols}


# ---------------- compare ----------------
def run_compare(exchange, symbols, tf="1m", approaches=None, n_folds=4, **kw) -> dict:
    o = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another approaches job is running")
    try:
        COMPARE.clear(); COMPARE.update(state="building dataset", started=int(time.time()))
        frames = load_frames(exchange, symbols, tf)
        data = evaluate.build_dataset(frames, o["horizon"], o["cost_bps"], o["k"])
        if data.empty:
            raise ValueError(f"no candles in SQLite for {exchange} {symbols}")
        COMPARE["state"] = "comparing"
        out = evaluate.compare(data, o["horizon"], o["cost_bps"], approaches, n_folds, o["thr"], o["gate_mult"],
                               progress=lambda s: COMPARE.update(detail=s))
        out["volatility"] = vol.score_vol({s: d for s, d in frames.items() if len(d) >= 200}, o["horizon"])
        out["config"] = dict(exchange=exchange, symbols=symbols, tf=tf, n_folds=n_folds, **o)
        COMPARE.clear(); COMPARE.update(state="done", finished=int(time.time()), result=out)
        return out
    except Exception as e:
        COMPARE.clear(); COMPARE.update(state="error", error=str(e))
        raise
    finally:
        _lock.release()


# ---------------- final model ----------------
def train_final(approach, exchange, symbols, tf="1m", cal_frac=0.2, **kw) -> dict:
    """Fit on the older (1-cal_frac) of the data, fit an isotonic calibrator on the newest cal_frac (after a purge gap), save a bundle."""
    o = {**DEFAULTS, **{k: v for k, v in kw.items() if v is not None}}
    cls = models.REGISTRY[approach]
    if not cls.available():
        raise RuntimeError(f"{approach} needs `pip install {cls.needs}`")
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another approaches job is running")
    try:
        TRAIN.clear(); TRAIN.update(state="training", approach=approach, started=int(time.time()))
        data = evaluate.build_dataset(load_frames(exchange, symbols, tf), o["horizon"], o["cost_bps"], o["k"])
        lab = data[data["y"].notna()]
        if len(lab) < evaluate.MIN_LABELED:
            raise ValueError(f"only {len(lab)} labeled rows; need >= {evaluate.MIN_LABELED}")
        cut = lab["ts"].quantile(1 - cal_frac)
        tr = lab[lab["ts"] + o["horizon"] * MIN_MS <= cut]
        cal = lab[lab["ts"] >= cut]
        model = models.make(approach).fit(tr[FEATURES], tr["y"].astype(int).to_numpy())
        calibrator, holdout_auc = None, None
        if len(cal) >= 200 and cal["y"].nunique() == 2:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import roc_auc_score
            raw = model.predict_proba(cal[FEATURES])
            holdout_auc = float(roc_auc_score(cal["y"], raw))
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(raw, cal["y"].to_numpy())
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        path = str(config.MODEL_DIR / f"approach_{approach}_{stamp}.joblib")
        meta = dict(approach=approach, exchange=exchange, symbols=symbols, tf=tf, horizon=o["horizon"], cost_bps=o["cost_bps"], k=o["k"],
                    thr=o["thr"], gate_mult=o["gate_mult"], n_train=int(len(tr)), n_cal=int(len(cal)), calibrated=calibrator is not None,
                    holdout_auc=holdout_auc, trained_ts=stamp * 1000, path=path)
        dump(dict(model=model, calibrator=calibrator, features=FEATURES, meta=meta), path)
        _registry_path().write_text(json.dumps(_registry() + [meta], indent=1))
        TRAIN.clear(); TRAIN.update(state="done", **{k: meta[k] for k in ("approach", "n_train", "n_cal", "calibrated", "holdout_auc", "path")})
        return meta
    except Exception as e:
        TRAIN.clear(); TRAIN.update(state="error", error=str(e))
        raise
    finally:
        _lock.release()


def saved_models() -> list[dict]:
    return _registry()


def _load_bundle(approach: str | None):
    reg = [m for m in _registry() if approach in (None, m["approach"])]
    if not reg:
        raise LookupError("no saved model" + (f" for {approach!r}" if approach else "") + " - POST /api/ml/approaches/train first")
    return load(reg[-1]["path"])


# ---------------- decision layer ----------------
def decide(p_up: float, sigma_bar: float, horizon: int, cost_bps: float, thr: float, gate_mult: float) -> dict:
    """Turn a probability + a volatility forecast into long / short / no_trade with a reason.
    expected_move_bps = 0.8 * sigma * sqrt(H)   (E|N(0,s)| = 0.8 s)      edge_bps = (2p-1) * expected_move_bps   <- a HEURISTIC, not a forecast."""
    move = 0.8 * sigma_bar * np.sqrt(horizon) * 1e4
    edge = (2 * p_up - 1) * move
    side = "long" if p_up >= thr else ("short" if p_up <= 1 - thr else None)
    if side is None:
        action, reason = "no_trade", f"probability {p_up:.3f} is inside the neutral band {1 - thr:.2f}-{thr:.2f}"
    elif move < gate_mult * cost_bps:
        action, reason = "no_trade", f"expected move {move:.1f} bps < {gate_mult}x cost ({cost_bps:.1f} bps): too quiet to pay for the trade"
    elif abs(edge) < cost_bps:
        action, reason = "no_trade", f"heuristic edge {abs(edge):.1f} bps does not cover cost {cost_bps:.1f} bps"
    else:
        action, reason = side, f"p_up {p_up:.3f}, heuristic edge {abs(edge):.1f} bps vs cost {cost_bps:.1f} bps"
    return dict(trade_action=action, reason=reason, expected_move_bps=round(float(move), 2), heuristic_edge_bps=round(float(edge), 2))


def _regime(row: pd.Series, sigma: pd.Series) -> str:
    med = sigma.tail(500).median()
    v = "high" if sigma.iloc[-1] > 1.5 * med else ("low" if sigma.iloc[-1] < 0.67 * med else "normal")
    t = "up" if row["ema_gap_21"] > 0 else "down"
    return f"{v}_vol_{t}trend"


def predict(exchange: str, symbol: str, tf: str = "1m", approach: str | None = None) -> dict:
    b = _load_bundle(approach)
    m = b["meta"]
    d = to_frame(db.dataset(exchange, symbol, tf)).tail(400).reset_index(drop=True)
    if len(d) < 80:
        raise LookupError(f"need >= 80 candles for {exchange} {symbol}, have {len(d)}")
    f = build_features(d)
    row = f.iloc[-1]
    if row.isna().any():
        raise LookupError("latest candle has incomplete features (warm-up or a gap in the last 60 minutes)")
    p = float(b["model"].predict_proba(f.iloc[[-1]])[0])
    if b["calibrator"] is not None:
        p = float(b["calibrator"].predict([p])[0])
    sigma = vol.ewma(vol.log_returns(d["close"]))
    out = dict(exchange=exchange, symbol=symbol, timestamp=int(d["ts"].iloc[-1]), data_age_s=int(time.time() - d["ts"].iloc[-1] / 1000),
               horizon_minutes=m["horizon"], direction_probability_up=round(p, 4), calibrated=m["calibrated"],
               predicted_vol_bps_per_bar=round(float(sigma.iloc[-1] * 1e4), 3), market_regime=_regime(row, sigma),
               approach=m["approach"], model_trained_ts=m["trained_ts"])
    out.update(decide(p, float(sigma.iloc[-1]), m["horizon"], m["cost_bps"], m["thr"], m["gate_mult"]))
    return out

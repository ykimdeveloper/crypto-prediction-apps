"""Optional direction model: h2o AutoML trained on the indicator rows in SQLite, persisted with joblib.

  features  : where price sits in the Bollinger tunnel, band width, Stoch RSI (k, d, slope), ATR % of price, 1/5/15-candle returns, volume ratio
  label     : did the close, `horizon` candles later (default 15 x 1m = one contract), finish above this candle's close?
  output    : ml_p_up (0..1) written to indicators.ml_p_up and shown on the card. It is INFORMATION ONLY - it never gates a strategy.

joblib does two jobs: Parallel() builds the per-symbol feature tables, and dump()/load() stores the model bundle
(feature list, horizon, metrics, path of the h2o binary). h2o itself needs a Java runtime; if h2o isn't installed the
rest of the server runs fine and /api/ml/train answers 501.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load

from . import config, db

FEATURES = ["bb_pos", "bb_width", "stoch_k", "stoch_d", "stoch_delta", "atr_pct", "ret_1", "ret_5", "ret_15", "vol_ratio"]
MIN_ROWS = 500
STATUS: dict = {"state": "idle"}
_train_lock = threading.Lock()


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"c": "close", "v": "volume", "o": "open", "h": "high", "l": "low"})


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = _norm(df).copy()
    up = (d["bb_upper"] - d["bb_mid"]).replace(0, np.nan)
    f = pd.DataFrame(index=d.index)
    f["bb_pos"] = (d["close"] - d["bb_mid"]) / up
    f["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"]
    f["stoch_k"], f["stoch_d"], f["stoch_delta"] = d["stoch_k"], d["stoch_d"], d["stoch_delta"]
    f["atr_pct"] = d["atr"] / d["close"]
    for n in (1, 5, 15):
        f[f"ret_{n}"] = d["close"].pct_change(n)
    f["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean().replace(0, np.nan)
    return f.replace([np.inf, -np.inf], np.nan)


def _build_symbol(rows: list[dict], horizon: int) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    f = make_features(df)
    f["y"] = (df["close"].shift(-horizon) > df["close"]).astype(int).where(df["close"].shift(-horizon).notna())
    f["ts"] = df["ts"]
    return f.dropna()


def build_dataset(exchange: str, symbols: list[str], tf: str, horizon: int) -> pd.DataFrame:
    per_symbol = [db.dataset(exchange, s, tf) for s in symbols]           # sqlite reads stay on this thread
    jobs = min(len(per_symbol), os.cpu_count() or 1) or 1
    parts = Parallel(n_jobs=jobs)(delayed(_build_symbol)(rows, horizon) for rows in per_symbol)
    parts = [p.assign(symbol=s) for p, s in zip(parts, symbols) if len(p)]
    return pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True) if parts else pd.DataFrame()


def _h2o():
    import h2o  # noqa: WPS433  (optional dependency)
    h2o.init(nthreads=-1, max_mem_size=os.getenv("H2O_MAX_MEM", "2G"), verbose=False)
    return h2o


def train(exchange: str, symbols: list[str], tf: str = "1m", horizon: int | None = None, max_runtime_secs: int = 120, max_models: int = 8) -> dict:
    """Blocking (run it in a thread). Time-ordered 80/20 split: the model is scored on data that comes AFTER what it trained on."""
    horizon = horizon or config.ML_HORIZON
    if not _train_lock.acquire(blocking=False):
        raise RuntimeError("a training run is already in progress")
    try:
        STATUS.clear(); STATUS.update(state="building dataset", started=int(time.time()))
        data = build_dataset(exchange, symbols, tf, horizon)
        if len(data) < MIN_ROWS:
            raise ValueError(f"only {len(data)} labeled rows in SQLite for {exchange} {symbols}; need >= {MIN_ROWS}. Let the feed run longer.")
        STATUS["state"] = "training"
        h2o = _h2o()
        from h2o.automl import H2OAutoML
        cut = int(len(data) * 0.8)
        cols = FEATURES + ["y"]
        tr = h2o.H2OFrame(data.iloc[:cut][cols]); va = h2o.H2OFrame(data.iloc[cut:][cols])
        tr["y"] = tr["y"].asfactor(); va["y"] = va["y"].asfactor()
        aml = H2OAutoML(max_models=max_models, max_runtime_secs=max_runtime_secs, seed=1, sort_metric="AUC", nfolds=0,
                        exclude_algos=["StackedEnsemble", "DeepLearning"])
        aml.train(x=FEATURES, y="y", training_frame=tr, validation_frame=va, leaderboard_frame=va)
        leader = aml.leader
        perf = leader.model_performance(va)
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        model_path = h2o.save_model(model=leader, path=str(config.MODEL_DIR / f"h2o_{stamp}"), force=True)
        bundle_path = str(config.MODEL_DIR / f"bundle_{stamp}.joblib")
        info = dict(features=FEATURES, horizon=horizon, tf=tf, exchange=exchange, symbols=symbols, model_path=model_path,
                    algo=leader.algo, auc=float(perf.auc()), logloss=float(perf.logloss()), n_rows=len(data), h2o_version=h2o.__version__,
                    trained_ts=stamp * 1000)
        dump(info, bundle_path)
        db.save_model(created_ts=stamp * 1000, exchange=exchange, symbols=",".join(symbols), tf=tf, horizon=horizon, n_rows=len(data),
                      algo=leader.algo, auc=info["auc"], logloss=info["logloss"], bundle_path=bundle_path)
        PREDICTOR.reset()
        STATUS.clear(); STATUS.update(state="done", **{k: info[k] for k in ("algo", "auc", "logloss", "n_rows", "horizon")}, bundle=bundle_path)
        return dict(STATUS)
    except Exception as e:
        STATUS.clear(); STATUS.update(state="error", error=str(e))
        raise
    finally:
        _train_lock.release()


class Predictor:
    """Lazy: loads the newest active bundle the first time it's asked, and again after a retrain."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.bundle, self.model, self._tried = None, None, False

    def _load(self):
        self._tried = True
        m = db.active_model()
        if not m or not Path(m["bundle_path"]).exists():
            return
        try:
            self.bundle = load(m["bundle_path"])
            self.model = _h2o().load_model(self.bundle["model_path"])
        except Exception as e:  # missing h2o / Java / version mismatch: run without ML rather than break the feed
            print(f"[ml] model not loaded: {e}")
            self.bundle = self.model = None

    def p_up(self, frame: pd.DataFrame) -> float | None:
        """frame = Engine.frame (closed candles + indicator columns). Blocking - call from a worker thread."""
        with self._lock:
            if not self._tried:
                self._load()
            if self.model is None or len(frame) < 30:
                return None
            row = make_features(frame).iloc[[-1]][self.bundle["features"]]
            if row.isna().any(axis=None):
                return None
            h2o = _h2o()
            out = self.model.predict(h2o.H2OFrame(row)).as_data_frame()
            return float(out["p1"].iloc[0])


PREDICTOR = Predictor()


def status() -> dict:
    return dict(training=dict(STATUS), model=db.active_model(), h2o_available=_has_h2o(), enabled=config.ML_ENABLED)


def _has_h2o() -> bool:
    import importlib.util
    return importlib.util.find_spec("h2o") is not None

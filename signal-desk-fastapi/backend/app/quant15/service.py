"""compare / train / decide. Blocking: the routes run them in worker threads (BackgroundTasks / to_thread), like approaches.service."""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np
import pandas as pd
from joblib import dump, load

from .. import config, db
from ..constants import MIN_MS, PAPER, STRAT_KEYS
from . import bars as B, combine, model15 as M, replay as R, vol15 as V
from .features15 import FEATURES15, build_features15

STALE_S = 150          # /decide refuses to act on 1m candles older than this (live Engine uses 20s; minute candles need slack)
COMPARE: dict = {"state": "idle"}
TRAIN: dict = {"state": "idle"}
_lock = threading.Lock()


def _safe(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x)


def _bundle_path(exchange: str):
    return config.MODEL_DIR / f"quant15_{_safe(exchange)}.joblib"


def _policy_path():
    return config.MODEL_DIR / "quant15_policy.json"


def _load_frames(exchange: str, symbols: list[str]):
    bars = {s: B.load_15m(exchange, s) for s in symbols}
    peers = {}
    for s in symbols:
        ps = M.peer_symbol(s)
        if ps not in bars and ps not in peers:
            pb = B.load_15m(exchange, ps)
            if not pb.empty:
                peers[ps] = pb
    return bars, {**{k: v for k, v in bars.items()}, **peers}


# ---------------------------------------------------------------- compare
def run_compare(exchange: str, symbols: list[str], approaches: list[str] | None = None, n_folds: int = 4,
                margin: float = 0.02, vol_method: str = "har") -> dict:
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another quant15 job is running")
    try:
        COMPARE.clear(); COMPARE.update(state="building dataset", started=int(time.time()))
        bars, peers = _load_frames(exchange, symbols)
        data = M.build_dataset15(bars, peers)
        if data.empty:
            raise ValueError(f"need >= {M.MIN_BARS} 15m bars per symbol; have {({s: len(b) for s, b in bars.items()})}. "
                             "POST /api/quant15/backfill first (tf=1m for ~30-90 days, or tf=15m for longer history).")
        names = M.pick_models(approaches)
        COMPARE["state"] = f"walk-forward: {', '.join(names)}"
        preds, skipped = M.oof(data, names, n_folds)
        mets = M.metrics(data, preds)
        ens = next((r for r in mets if r["approach"] == "ensemble"), {})
        test_start = int(data.loc[~np.isnan(preds["ensemble"]), "ts"].min())
        ens_p = pd.Series(preds["ensemble"], index=data.index)

        COMPARE["state"] = "replaying the original algorithm on 1m candles"
        sigs, wins, steps, prior, sig15, n_1m = [], [], [], {}, {}, {}
        K = lambda sym, W: int(W) * 100 + symbols.index(sym)   # one integer key per (symbol, window), still ordered by time
        for sym in symbols:
            rows = data[(data["symbol"] == sym) & ens_p.notna()]
            if rows.empty:
                continue
            for ts, p in zip(rows["ts"], ens_p[rows.index]):
                prior[K(sym, int(ts) + B.BAR_MS)] = float(p)
            b = bars[sym]
            lr = V.log_returns(b["close"])
            fit_end = int((b["ts"] < test_start).sum())
            s15 = V.forecast(lr, fit_end, vol_method)
            sig15.update({K(sym, int(t) + B.BAR_MS): float(v) for t, v in zip(b["ts"], s15) if pd.notna(v)})
            d1 = B.load_1m(exchange, sym)
            n_1m[sym] = int(len(d1))
            d1 = d1[d1["ts"] >= test_start + B.BAR_MS - 200 * MIN_MS]         # 200 minutes of indicator warm-up before the first test window
            rp = R.replay(d1)
            for x in rp["signals"]:
                x["symbol"], x["win_start"] = sym, K(sym, x["win_start"]); sigs.append(x)
            for w in rp["windows"]:
                w["symbol"], w["win_start"] = sym, K(sym, w["win_start"]); wins.append(w)
            st = rp["steps"].copy(); st["win_start"] = [K(sym, w) for w in st["win_start"]]; steps.append(st)

        report = dict(config=dict(exchange=exchange, symbols=symbols, models=names, n_folds=n_folds, margin=margin, vol_method=vol_method),
                      rows=int(len(data)), labeled=int(data["y"].notna().sum()), model=mets, skipped=skipped, one_minute_rows=n_1m,
                      warnings=[])
        if not sigs:
            report["warnings"].append("No original-algorithm signals in the out-of-sample period: backfill 1m candles (tf=1m) so the replay has data.")
            report["policies"] = None; report["sigma_calibration"] = None
            report["verdict"] = dict(policy="original", combine=False, reasons=["no replayed signals to compare against"])
        else:
            pol = combine.evaluate_policies(sigs, wins, prior, margin)
            report["policies"] = pol
            steps_df = pd.concat(steps, ignore_index=True)
            report["sigma_calibration"] = combine.sigma_calibration(steps_df, sig15)
            report["verdict"] = combine.verdict(pol, ens.get("auc"), ens.get("n"), float(data["y"].mean()))
            if pol["original"]["n"] < combine.MIN_TRADES:
                report["warnings"].append(f"only {pol['original']['n']} original-algorithm trades in the test period; treat every number as anecdotal.")
        if ens.get("auc") is not None and ens["auc"] < combine.MIN_AUC:
            report["warnings"].append("Ensemble AUC < 0.52: no detectable direction edge at 15 minutes. The volatility layer may still be useful; the direction gate is not.")
        report["warnings"].append("Cents are paper-style PnL at the MODELED book (strategy.book_model), not real quotes. Hit rates use real settlement outcomes.")
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _policy_path().write_text(json.dumps(dict(exchange=exchange, ts=int(time.time()), margin=margin, **report["verdict"])))
        COMPARE.clear(); COMPARE.update(state="done", finished=int(time.time()), result=report)
        return report
    except Exception as e:
        COMPARE.clear(); COMPARE.update(state="error", error=str(e))
        raise
    finally:
        _lock.release()


# ---------------------------------------------------------------- train
def train_final(exchange: str, symbols: list[str], approaches: list[str] | None = None, cal_frac: float = 0.2,
                vol_method: str = "har") -> dict:
    if not _lock.acquire(blocking=False):
        raise RuntimeError("another quant15 job is running")
    try:
        TRAIN.clear(); TRAIN.update(state="training", started=int(time.time()))
        bars, peers = _load_frames(exchange, symbols)
        data = M.build_dataset15(bars, peers)
        lab = data[data["y"].notna()] if not data.empty else data
        if len(lab) < 500:
            have = {sym: dict(candles_1m=int(len(B.load_1m(exchange, sym))), bars_15m=int(len(bars[sym]))) for sym in symbols}
            raise ValueError(f"only {len(lab)} labeled 15m rows; need >= 500 and >= {M.MIN_BARS} 15m bars per symbol. Stored for '{exchange}': {have}. "
                             "A 15m bar needs >= 13 of its 15 one-minute candles, so a thinly traded pair (e.g. Coinbase BTC/USDT) yields almost none: "
                             "train on the exact exchange label + symbol the desk records (see GET /api/quant15/data), e.g. coinbaseexchange BTC/USD.")
        names = M.pick_models(approaches)
        cut = lab["ts"].quantile(1 - cal_frac)
        tr, cal = lab[lab["ts"] + M.LABEL_BARS * B.BAR_MS <= cut], lab[lab["ts"] >= cut]
        models = M.fit_ensemble(tr[FEATURES15], tr["y"].astype(int).to_numpy(), names)
        calibrator, holdout_auc = None, None
        if len(cal) >= 200 and cal["y"].nunique() == 2:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.metrics import roc_auc_score
            raw = M.predict_ensemble(models, cal[FEATURES15])
            holdout_auc = float(roc_auc_score(cal["y"], raw))
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95).fit(raw, cal["y"].to_numpy())
        volm = {}
        for s in symbols:
            b = bars[s]
            if len(b) < M.MIN_BARS:
                continue
            lr = V.log_returns(b["close"])
            har = V.HAR15().fit(lr[b["ts"] < cut])
            sig = har.predict(lr)
            volm[s] = dict(har=[float(x) for x in har.coef], edges=V.edges(sig[b["ts"] < cut]))
        stamp = int(time.time())
        meta = dict(exchange=exchange, symbols=symbols, models=names, n_train=int(len(tr)), n_cal=int(len(cal)),
                    calibrated=calibrator is not None, holdout_auc=holdout_auc, trained_ts=stamp * 1000, vol_method=vol_method)
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dump(dict(models=models, calibrator=calibrator, features=FEATURES15, vol=volm, meta=meta), _bundle_path(exchange))
        TRAIN.clear(); TRAIN.update(state="done", **{k: meta[k] for k in ("models", "n_train", "n_cal", "calibrated", "holdout_auc")})
        return meta
    except Exception as e:
        TRAIN.clear(); TRAIN.update(state="error", error=str(e))
        raise
    finally:
        _lock.release()


# ---------------------------------------------------------------- decide (live)
def prior_now(bundle: dict, exchange: str, symbol: str, now_ms: int | None = None) -> dict:
    """Model view for the window that is in progress: P(UP) from the last CLOSED 15m bar, vol forecast and regime."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    W = now_ms // B.BAR_MS * B.BAR_MS
    b = B.load_15m(exchange, symbol, now_ms=now_ms).tail(900).reset_index(drop=True)
    pb = B.load_15m(exchange, M.peer_symbol(symbol), now_ms=now_ms).tail(900)
    if len(b) < 700 or int(b["ts"].iloc[-1]) != W - B.BAR_MS:
        return dict(available=False, reason=f"need >= 700 contiguous closed 15m bars ending at the previous window; have {len(b)}, last={None if b.empty else int(b['ts'].iloc[-1])}")
    f = build_features15(b, pb if not pb.empty else None)
    row = f.iloc[[-1]]
    if row.isna().all(axis=1).iloc[0]:
        return dict(available=False, reason="latest features are all NaN (gap in the last day of bars)")
    p = float(M.predict_ensemble(bundle["models"], row)[0])
    if bundle["calibrator"] is not None:
        p = float(bundle["calibrator"].predict([p])[0])
    out = dict(available=True, window_start=W, p_up=round(p, 4), models=list(bundle["models"]))
    v = bundle["vol"].get(symbol)
    if v:
        lr = V.log_returns(b["close"])
        sig = float(V.HAR15(v["har"]).predict(lr).iloc[-1])
        out.update(sigma_bps_per_15m_bar=round(sig * 1e4, 2), regime=V.regime(sig, v["edges"]))
    return out


def decide(exchange: str, symbol: str, policy: str | None = None, margin: float | None = None, now_ms: int | None = None) -> dict:
    """Original signals + model prior + (optionally) the gated decision. policy: original | gated | model_only.
    Default policy = what the last /compare run concluded, else 'original' (i.e. nothing changes until evidence says so)."""
    pol_file = json.loads(_policy_path().read_text()) if _policy_path().exists() else {}
    policy = policy or pol_file.get("policy", "original")
    margin = margin if margin is not None else pol_file.get("margin", 0.02)
    d1 = B.load_1m(exchange, symbol).tail(400)
    if len(d1) < 60:
        raise LookupError(f"need >= 60 stored 1m candles for {exchange} {symbol}, have {len(d1)} (let the feed run or backfill)")
    age_s = int(time.time() - d1["ts"].iloc[-1] / 1000)
    sig = R.signals_now(d1)
    try:
        bundle = load(_bundle_path(exchange))
        model = prior_now(bundle, exchange, symbol, now_ms)
    except FileNotFoundError:
        model = dict(available=False, reason="no trained quant15 model: POST /api/quant15/train first")
    out = dict(exchange=exchange, symbol=symbol, data_age_s=age_s, policy=policy, margin=margin, model=model, module=sig.get("module"),
               rem_min=sig.get("remMin"), strategies={})
    for s in STRAT_KEYS:
        cur = sig[s]
        buy = cur["signal"].startswith("BUY") and (cur["conf"] or 0) >= PAPER["minConf"]
        d = "UP" if cur["signal"].endswith("UP") else "DOWN"
        row = dict(original=cur["signal"], conf=cur["conf"], ask=cur.get("ask"), reason=cur["reason"], final=cur["signal"], note=None)
        if buy and age_s > STALE_S:
            row.update(final="WAIT", note=f"stale: newest 1m candle is {age_s}s old")
        elif buy and policy == "gated":
            if not model["available"]:
                row["note"] = f"model unavailable ({model['reason']}): original passes through"
            elif combine.agrees(d, model["p_up"], margin):
                row["note"] = f"model agrees (p_up {model['p_up']:.3f})"
            else:
                row.update(final="WAIT", note=f"VETO: model p_up {model['p_up']:.3f} disagrees with {d}")
        out["strategies"][s] = row
    if model.get("available") and abs(model["p_up"] - 0.5) >= margin:
        out["model_only"] = dict(side="UP" if model["p_up"] > 0.5 else "DOWN", p_up=model["p_up"],
                                 note="window-start view only; the entry price must come from the real market")
    return out


def status() -> dict:
    return dict(compare=COMPARE, train=TRAIN, policy=json.loads(_policy_path().read_text()) if _policy_path().exists() else None)


def inventory(exchange: str | None = None) -> dict:
    """What the database actually holds, per (exchange label, symbol), 1m candles only, plus which labels have a trained model.
    coverage = stored candles / minutes between first and last (1.0 = no gaps); usable_15m_bars = windows with >= 13 of 15 candles,
    which is what training and the live gate really see."""
    q = "FROM candles WHERE tf='1m'" + (" AND exchange=?" if exchange else "")
    a = (exchange,) if exchange else ()
    with db._lock:
        rows = db._conn.execute(f"SELECT exchange,symbol,COUNT(*) n,MIN(ts) a,MAX(ts) b {q} GROUP BY exchange,symbol ORDER BY exchange,symbol", a).fetchall()
        use = {(r[0], r[1]): r[2] for r in db._conn.execute(
            f"SELECT exchange,symbol,COUNT(*) FROM (SELECT exchange,symbol,ts/{B.BAR_MS} w FROM candles WHERE tf='1m'"
            + (" AND exchange=?" if exchange else "") + " GROUP BY exchange,symbol,w HAVING COUNT(*)>=13) GROUP BY exchange,symbol", a).fetchall()}
    now = int(time.time() * 1000)
    out = []
    for ex, sym, n, lo, hi in rows:
        span = (hi - lo) // MIN_MS + 1
        bars15 = use.get((ex, sym), 0)
        out.append(dict(exchange=ex if len(ex) < 40 else ex[:37] + "...", symbol=sym, candles_1m=n, coverage=round(n / span, 3),
                        first=time.strftime("%F %T", time.gmtime(lo / 1000)), last=time.strftime("%F %T", time.gmtime(hi / 1000)),
                        last_age_h=round((now - hi) / 3.6e6, 1), usable_15m_bars=bars15,
                        can_train=bars15 >= M.MIN_BARS, has_model=_bundle_path(ex).exists()))
    return dict(rows=out, note="can_train needs usable_15m_bars >= 800 (about 8.3 days of gap-free 1m). Live gating additionally needs the last ~700 bars contiguous and current.")

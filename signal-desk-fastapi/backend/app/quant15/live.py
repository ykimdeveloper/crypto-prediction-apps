"""Live signal policy for the desk: original | gated.

  original  the desk behaves exactly as before (default; nothing here runs).
  gated     a rev/mom/end BUY is kept only if the trained 15m model's P(UP) for the current window agrees with its direction
            by >= margin. Otherwise it becomes WAIT with the reason. Applied BEFORE paper trading, so the ledger, alerts and the
            signal log all follow the policy.
  (model_only exists in /compare and /decide as research; it is not offered live, because a paper entry needs a real market price.)

The model is asked once per 15m window, in a worker thread, and cached; the 1-second desk loop never waits for it.
While the answer is pending, BUYs are held (a veto policy must not let a trade slip through before the model has spoken).
If no model can be used (not trained for this exchange label, too little data), signals pass through UNCHANGED and the card says so.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from joblib import load

from .. import config
from ..constants import PAPER, STRAT_KEYS
from . import bars as B, combine, service as S

POLICIES = ("original", "gated")
RETRY_UNAVAILABLE_S = 10        # the previous window's last 1m candle may land a second after the window rolls
_cfg: dict | None = None
_lock = threading.Lock()
_cache: dict[tuple[str, str], dict] = {}
_bundles: dict[str, tuple[float, dict]] = {}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="quant15-live")


def _path():
    return config.MODEL_DIR / "quant15_live_policy.json"


def get() -> dict:
    global _cfg
    with _lock:
        if _cfg is None:
            try:
                d = json.loads(_path().read_text())
                _cfg = dict(policy=d["policy"] if d.get("policy") in POLICIES else "original", margin=float(d.get("margin", 0.02)))
            except (OSError, ValueError, KeyError):
                _cfg = dict(policy="original", margin=0.02)
        return dict(_cfg)


def set_policy(policy: str, margin: float | None = None) -> dict:
    global _cfg
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    cur = get()
    new = dict(policy=policy, margin=cur["margin"] if margin is None else float(margin))
    with _lock:
        _cfg = new
        _cache.clear()
    try:
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(new))
    except OSError as e:
        print(f"[quant15] could not persist policy: {e}")
    return new


def status() -> dict:
    ver = None
    try:
        ver = json.loads(S._policy_path().read_text())
    except (OSError, ValueError):
        pass
    trained = sorted(p.stem.removeprefix("quant15_") for p in config.MODEL_DIR.glob("quant15_*.joblib")) if config.MODEL_DIR.exists() else []
    return dict(**get(), options=list(POLICIES), trained=trained, compare=ver)


def _bundle(exchange: str) -> dict:
    p = S._bundle_path(exchange)
    m = p.stat().st_mtime          # FileNotFoundError when the exchange has no trained model
    hit = _bundles.get(exchange)
    if hit is None or hit[0] != m:
        hit = (m, load(p))
        _bundles[exchange] = hit
    return hit[1]


def _compute(exchange: str, symbol: str, win: int, now_ms: int):
    key = (exchange, symbol)
    try:
        try:
            prior = S.prior_now(_bundle(exchange), exchange, symbol, now_ms)
        except FileNotFoundError:
            prior = dict(available=False, reason=f"no trained model for '{exchange}': train on the exchange the desk is using (POST /api/quant15/train)")
    except Exception as e:  # noqa: BLE001  a broken model must never take the desk down
        prior = dict(available=False, reason=f"{type(e).__name__}: {e}")
    with _lock:
        _cache[key] = dict(win=win, prior=prior, at=time.time(), busy=False)


def prior(exchange: str, symbol: str, now_ms: int) -> dict | None:
    """Cached model view of the current window, or None while it is being computed."""
    win = now_ms // B.BAR_MS * B.BAR_MS
    key = (exchange, symbol)
    with _lock:
        c = _cache.get(key)
        fresh = c and c["win"] == win and (c["busy"] or c["prior"].get("available") or time.time() - c["at"] < RETRY_UNAVAILABLE_S)
        if fresh:
            return None if c["busy"] else c["prior"]
        _cache[key] = dict(win=win, prior=None, at=time.time(), busy=True)
    _pool.submit(_compute, exchange, symbol, win, now_ms)
    return None


def apply(exchange: str, symbol: str, sig: dict, now_ms: int) -> dict:
    """Mutates `sig` (vetoed BUYs become WAIT) and returns what the card should show about the policy."""
    cfg = get()
    info = dict(mode=cfg["policy"], active=False, pUp=None, note=None, vetoed=[], margin=cfg["margin"])
    if cfg["policy"] != "gated":
        return info
    pr = prior(exchange, symbol, now_ms)
    pending = pr is None
    if pending:
        info["note"] = "waiting for the model's view of this window; BUYs are held"
    elif not pr.get("available"):
        info["note"] = f"model unavailable, original signals pass through unchanged ({pr.get('reason')})"
        return info
    else:
        info.update(active=True, pUp=pr["p_up"])
    for s in STRAT_KEYS:
        cur = sig.get(s)
        if not cur or not str(cur.get("signal", "")).startswith("BUY") or (cur.get("conf") or 0) < PAPER["minConf"]:
            continue
        d = "UP" if cur["signal"].endswith("UP") else "DOWN"
        if pending:
            why = "waiting for the model"
        elif combine.agrees(d, pr["p_up"], cfg["margin"]):
            continue
        else:
            why = f"VETO: model P(up) {pr['p_up']:.3f} does not back {d} by {cfg['margin']:.2f}"
        sig[s] = dict(signal="WAIT", conf=cur.get("conf"), reason=why, original=cur["signal"])
        info["vetoed"].append(s)
    return info

"""Server-side toggle + settings. Changing them never touches the websocket or the feeds."""
from __future__ import annotations

import json
import threading

from .. import config as app_config
from .data import SPAN_MS, TF_MS

WINDOW_CHOICES = (8, 24, 96, 288, 672)
SHOW = ("nearest", "all_in_move", "all")
DEFAULTS = dict(enabled=False, tfs=["5m", "15m", "1h"], spans=["1d", "1w", "1mo"], windows=96, min_touches=2,
                confluence_only=False, journal=True, show="nearest", horizon_min=15, x_atr=0.5, zone_width_atr=0.3, round_numbers=False)
_lock = threading.Lock()
_cfg: dict | None = None


def _path():
    return app_config.MODEL_DIR / "levels_config.json"


def validate(c: dict) -> dict:
    out = dict(DEFAULTS)
    out.update({k: c[k] for k in DEFAULTS if k in c})
    out["enabled"] = bool(out["enabled"])
    out["confluence_only"] = bool(out["confluence_only"])
    out["round_numbers"] = bool(out["round_numbers"])
    out["journal"] = bool(out["journal"])
    out["tfs"] = [t for t in out["tfs"] if t in TF_MS] or DEFAULTS["tfs"]
    out["spans"] = ["12mo" if s == "1y" else s for s in out["spans"]]
    out["spans"] = [s for s in out["spans"] if s in SPAN_MS] or DEFAULTS["spans"]
    if int(out["windows"]) < 1:
        raise ValueError("windows must be >= 1 (presets: 8, 24, 96, 288, 672, or any custom N)")
    out["windows"] = int(out["windows"])
    out["min_touches"] = max(0, int(out["min_touches"]))
    if out["show"] not in SHOW:
        raise ValueError(f"show must be one of {SHOW}")
    for k, lo in (("horizon_min", 1), ("x_atr", 0.05), ("zone_width_atr", 0.05)):
        out[k] = max(lo, float(out[k]))
    return out


def get() -> dict:
    global _cfg
    with _lock:
        if _cfg is None:
            try:
                _cfg = validate(json.loads(_path().read_text()))
            except (OSError, ValueError, TypeError):
                _cfg = dict(DEFAULTS)
        return json.loads(json.dumps(_cfg))


def update(patch: dict) -> dict:
    global _cfg
    new = validate({**get(), **patch})
    with _lock:
        _cfg = new
    try:
        app_config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(new))
    except OSError as e:
        print(f"[levels] could not persist config: {e}")
    return json.loads(json.dumps(new))

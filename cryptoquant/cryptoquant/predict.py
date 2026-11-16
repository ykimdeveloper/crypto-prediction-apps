import joblib
import numpy as np
from . import config, vol
from .data import get_bars
from .features import build_features
from .models import ensemble

_art = {}


def load(symbol):
    if symbol not in _art:
        _art[symbol] = joblib.load(config.ART_DIR / f"{symbol}.joblib")
    return _art[symbol]


def reload():
    _art.clear()


def predict_latest(symbol):
    art = load(symbol)
    bars = get_bars(symbol).tail(config.TAIL_BARS)
    other = get_bars(config.OTHER[symbol]).tail(config.TAIL_BARS)
    X = build_features(bars, other, art["garch"], art["har"])
    row = X.iloc[[-1]][art["features"]]

    probs = {k: float(m.predict_proba(row)[0, 1]) for k, m in art["models"].items()}
    p = float(ensemble(probs))
    hv = float(row["har_vol"].iloc[0])
    reg = vol.regime(hv, art["vol_q"])
    edge = p - 0.5
    signal = "flat" if abs(edge) < config.EDGE[reg] else ("long" if edge > 0 else "short")
    return {
        "symbol": symbol,
        "bar_time": str(row.index[0]),
        "horizon_bars": config.HORIZON,
        "prob_up": p,
        "model_probs": probs,
        "garch_sigma_per_bar": float(row["garch_sigma"].iloc[0]),
        "har_vol_per_bar": hv,
        "vol_regime": reg,
        "required_edge": config.EDGE[reg],
        "signal": signal,
    }

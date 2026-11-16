"""Features for one touch event (or one live zone about to be touched). Every value uses only what was known at the touch time t:
the zone as detected at its checkpoint, the touches before t whose outcome had already resolved, and candles up to t - 1."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import DAY, TF_MS

FEATURES = ["n_prior", "held_rate_prior", "mins_since_last", "age_min", "tf_ix", "span_log_days", "confluence", "width_atr", "vol_share",
            "dist_atr", "approach", "hour_sin", "hour_cos", "dow", "rv_20", "vol_regime", "n_sources", "is_pivot", "is_prior", "is_volume"]
TFS = list(TF_MS)
MIN_EFF = 30      # fewer independent observations than this -> edge 'unproven'


def vol_series(close: pd.Series):
    rv = np.log(close).diff().rolling(20).std()
    reg = rv / rv.rolling(200, min_periods=50).median()
    return rv, reg


def last(s: pd.Series):
    v = s.iloc[-1] if len(s) else np.nan
    return None if v != v else float(v)


def prior_stats(touches, t_ms, out_ms):
    """Touches strictly before t_ms; the held rate uses only those whose forward horizon had already closed by t_ms."""
    before = [x for x in touches if x["ts"] < t_ms]
    res = [x for x in before if x["ts"] + out_ms <= t_ms and x["outcome"] in ("held", "broke")]
    return dict(n=len(before), held_rate=(sum(x["outcome"] == "held" for x in res) / len(res)) if res else None,
                last_ts=before[-1]["ts"] if before else None)


def feature_row(z, ctx, approach, t_ms, prior, confluence, dist_atr, tf, span_ms):
    src = z["sources"]
    nan = float("nan")
    hour = ((t_ms // 3_600_000) % 24) + (t_ms % 3_600_000) / 3_600_000
    return dict(n_prior=prior["n"], held_rate_prior=prior["held_rate"] if prior["held_rate"] is not None else nan,
                mins_since_last=(t_ms - prior["last_ts"]) / 60_000 if prior["last_ts"] is not None else nan,
                age_min=(t_ms - z["scan_from"]) / 60_000, tf_ix=TFS.index(tf), span_log_days=math.log(span_ms / DAY), confluence=confluence,
                width_atr=z["width_atr"], vol_share=z["vol_share"], dist_atr=dist_atr, approach=1.0 if approach == "from_below" else -1.0,
                hour_sin=math.sin(2 * math.pi * hour / 24), hour_cos=math.cos(2 * math.pi * hour / 24), dow=float((t_ms // DAY + 4) % 7),
                rv_20=ctx.get("rv_20") if ctx.get("rv_20") is not None else nan, vol_regime=ctx.get("vol_regime") if ctx.get("vol_regime") is not None else nan,
                n_sources=len(src), is_pivot=float(any(s.startswith("pivot") for s in src)), is_prior=float(any(s.startswith("prev_") for s in src)),
                is_volume=float(any(s in ("volume_node", "round_number") for s in src)))

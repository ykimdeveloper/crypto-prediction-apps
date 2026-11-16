from __future__ import annotations

import pandas as pd

DAY = 86_400_000
TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}
SPAN_MS = {"1d": DAY, "2d": 2 * DAY, "3d": 3 * DAY, "4d": 4 * DAY, "5d": 5 * DAY, "1w": 7 * DAY, "2w": 14 * DAY, "3w": 21 * DAY,
           "1mo": 30 * DAY, "3mo": 90 * DAY, "4mo": 120 * DAY, "6mo": 180 * DAY, "8mo": 240 * DAY, "12mo": 365 * DAY, "2y": 730 * DAY}
SPAN_MS["1y"] = SPAN_MS["12mo"]   # "12 months" and "1 year" are the same span
COLS = ["ts", "open", "high", "low", "close", "volume"]


def resample(d1: pd.DataFrame, tf_ms: int, min_frac: float = 0.8) -> pd.DataFrame:
    """1m -> tf. A bucket with under `min_frac` of its 1m candles is dropped rather than guessed."""
    if d1.empty:
        return pd.DataFrame(columns=COLS)
    key = (d1["ts"].astype("int64") // tf_ms) * tf_ms
    out = d1.groupby(key).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                              volume=("volume", "sum"), n=("ts", "size"))
    out = out[out["n"] >= max(1, int(min_frac * tf_ms / 60_000))].drop(columns="n")
    out.index.name = "ts"
    return out.reset_index()


def load(exchange: str, symbol: str, tf: str) -> pd.DataFrame:
    """Native candles for `tf` first, then buckets resampled from stored 1m for anything the native rows do not cover."""
    from .. import db
    from ..approaches.features import to_frame
    native = to_frame(db.dataset(exchange, symbol, tf))
    if tf == "1m":
        return native if not native.empty else pd.DataFrame(columns=COLS)
    derived = resample(to_frame(db.dataset(exchange, symbol, "1m")), TF_MS[tf])
    parts = [p for p in (native, derived) if not p.empty]
    if not parts:
        return pd.DataFrame(columns=COLS)
    d = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="first").sort_values("ts").reset_index(drop=True)
    d["ts"] = d["ts"].astype("int64")
    return d

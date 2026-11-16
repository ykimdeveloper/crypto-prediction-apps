"""Candle-only features. Every value at row t uses rows <= t (tests/test_approaches.py truncates the future and checks).

Rows are NaN when the last LOOKBACK bars are not contiguous (feed outage / exchange gap), because a rolling window
straddling a hole silently mixes different time scales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import MIN_MS

LOOKBACK = 60
FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_30", "ret_60",
    "rv_5", "rv_15", "rv_30", "rv_60", "rv_ratio", "park_15", "atr_pct",
    "ema_gap_9", "ema_gap_21", "ema_gap_50", "ema_9_21", "rsi_14", "macd_hist",
    "vol_z", "vol_ratio", "body", "upper_wick", "lower_wick", "close_loc", "range_pos_60",
    "hour_sin", "hour_cos",
]


def to_frame(rows: list[dict]) -> pd.DataFrame:
    """db.dataset rows (ts, open, high, low, close, volume, ...) -> clean OHLCV frame sorted by ts."""
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d[["ts", "open", "high", "low", "close", "volume"]].drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.dropna()


def build_features(d: pd.DataFrame, tf_ms: int = MIN_MS) -> pd.DataFrame:
    """OHLCV frame -> feature frame with the same index. Also carries `ts` so callers can align."""
    o, h, l, c, v = (d[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    lr = lc.diff()
    f = pd.DataFrame(index=d.index)
    for n in (1, 3, 5, 15, 30, 60):
        f[f"ret_{n}"] = lc.diff(n)
    for n in (5, 15, 30, 60):
        f[f"rv_{n}"] = lr.rolling(n).std()
    f["rv_ratio"] = f["rv_5"] / f["rv_60"].replace(0, np.nan)               # short vs long vol: spike / calm detector
    f["park_15"] = np.sqrt((np.log(h / l) ** 2).rolling(15).mean() / (4 * np.log(2)))   # range-based (Parkinson) vol
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(15).mean() / c
    ema = {n: c.ewm(span=n, adjust=False, min_periods=n).mean() for n in (9, 21, 50)}
    for n, e in ema.items():
        f[f"ema_gap_{n}"] = c / e - 1
    f["ema_9_21"] = ema[9] / ema[21] - 1
    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))) / 50 - 1     # -1..1
    macd = c.ewm(span=12, adjust=False, min_periods=12).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / c
    vm, vs = v.rolling(30).mean(), v.rolling(30).std()
    f["vol_z"] = (v - vm) / vs.replace(0, np.nan)
    f["vol_ratio"] = v / vm.replace(0, np.nan)
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["close_loc"] = (c - l) / rng
    lo, hi = l.rolling(60).min(), h.rolling(60).max()
    f["range_pos_60"] = (c - lo) / (hi - lo).replace(0, np.nan)
    hod = ((d["ts"].astype("int64") // 60_000) % 1440) / 1440.0             # UTC time of day
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod)
    f = f.replace([np.inf, -np.inf], np.nan)
    broken = (d["ts"] - d["ts"].shift(LOOKBACK)) != LOOKBACK * tf_ms          # a hole somewhere in the last 60 bars
    f.loc[broken, FEATURES] = np.nan
    return f[FEATURES]

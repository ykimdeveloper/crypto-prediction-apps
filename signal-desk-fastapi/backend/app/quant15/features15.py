"""15m features. Row t uses bars <= t only (tests truncate the future and compare). Row t is the decision made at the START
of window t+1, so its label is the direction of bar t+1 (see model15.build_dataset15)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bars import BAR_MS

FEATURES15 = [
    "ret_1", "ret_2", "ret_4", "ret_8", "ret_16", "ret_32", "ret_96",
    "rv_4", "rv_16", "rv_96", "rv_672", "rv_ratio_4_96", "rv_ratio_96_672", "ewma_vol", "z_ret_1", "z_ret_4", "park_16",
    "body", "close_loc", "range_z", "rsi_14", "ema_gap_16", "ema_gap_96", "vol_z",
    "hod_sin", "hod_cos", "dow_sin", "dow_cos",
    "x_ret_1", "x_ret_4", "x_rv_16", "rel_ret_4",
]


def build_features15(b: pd.DataFrame, peer: pd.DataFrame | None = None) -> pd.DataFrame:
    o, h, l, c, v = (b[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r = lc.diff()
    f = pd.DataFrame(index=b.index)
    for n in (1, 2, 4, 8, 16, 32, 96):
        f[f"ret_{n}"] = lc.diff(n)
    for n in (4, 16, 96, 672):
        f[f"rv_{n}"] = np.sqrt((r ** 2).rolling(n).mean())
    f["rv_ratio_4_96"] = f["rv_4"] / f["rv_96"].replace(0, np.nan)
    f["rv_ratio_96_672"] = f["rv_96"] / f["rv_672"].replace(0, np.nan)
    ew = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False, min_periods=20).mean())
    f["ewma_vol"] = ew
    f["z_ret_1"] = r / ew.shift(1).replace(0, np.nan)
    f["z_ret_4"] = lc.diff(4) / (2 * ew.shift(1)).replace(0, np.nan)
    f["park_16"] = np.sqrt((np.log(h / l) ** 2).rolling(16).mean() / (4 * np.log(2)))
    rng = (h - l).replace(0, np.nan)
    f["body"] = (c - o) / rng
    f["close_loc"] = (c - l) / rng
    f["range_z"] = (h - l) / (h - l).rolling(96).mean().replace(0, np.nan)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["rsi_14"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))) / 50 - 1
    f["ema_gap_16"] = c / c.ewm(span=16, adjust=False, min_periods=16).mean() - 1
    f["ema_gap_96"] = c / c.ewm(span=96, adjust=False, min_periods=96).mean() - 1
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(672).mean()) / lv.rolling(672).std().replace(0, np.nan)
    ts = b["ts"].astype("int64")
    hod = ((ts // 60_000) % 1440) / 1440.0
    dow = ((ts // 86_400_000) + 3) % 7 / 7.0          # epoch day 0 was a Thursday
    f["hod_sin"], f["hod_cos"] = np.sin(2 * np.pi * hod), np.cos(2 * np.pi * hod)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)
    if peer is not None and len(peer):
        pc = peer.drop_duplicates("ts").set_index("ts")["close"].reindex(ts.to_numpy()).to_numpy()
        lp = pd.Series(np.log(pc), index=b.index)
        f["x_ret_1"], f["x_ret_4"] = lp.diff(1), lp.diff(4)
        f["x_rv_16"] = np.sqrt((lp.diff() ** 2).rolling(16).mean())
        f["rel_ret_4"] = f["ret_4"] - f["x_ret_4"]
    else:
        for k in ("x_ret_1", "x_ret_4", "x_rv_16", "rel_ret_4"):
            f[k] = np.nan
    f = f.replace([np.inf, -np.inf], np.nan)
    broken = (ts - ts.shift(96)) != 96 * BAR_MS       # a hole somewhere in the last day: rolling windows would mix time scales
    f.loc[broken, FEATURES15] = np.nan
    return f[FEATURES15]

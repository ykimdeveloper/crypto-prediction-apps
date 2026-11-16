"""Bollinger Bands, Stochastic RSI and ATR.

Default backend is the `ta` package (vectorised over the whole candle history, so the same call fills the SQLite
`indicators` table). If `ta` isn't installed, or INDICATOR_BACKEND=pandas, a numpy/pandas port of src/lib/indicators.js
is used instead; that one matches the browser math bit-for-bit and is what the parity test checks.

Known, small differences when using `ta` (they wash out after ~100 candles):
  * RSI is seeded from the first value (ewm adjust=False) instead of a 14-bar simple average.
  * When RSI is flat for 14 bars the Stoch RSI is 0/0; the JS app calls that 50, `ta` gives NaN (handled below as 50).
The JS quirk "RSI = 99.0 when there are no losses" (instead of 100) is only reproduced by the pandas backend.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config
from .constants import (ATR_FLOOR_BPS_PER_MIN, ATR_PERIOD, BB_PERIOD, MIN_MS, RSI_PERIOD, STOCH_PERIOD)

try:  # pragma: no cover - depends on the environment
    from ta.momentum import StochRSIIndicator
    from ta.volatility import AverageTrueRange, BollingerBands
    HAVE_TA = True
except Exception:  # pragma: no cover
    HAVE_TA = False

COLS = ["t", "o", "h", "l", "c", "v"]
IND_COLS = ["bb_mid", "bb_upper", "bb_lower", "bb_state", "stoch_k", "stoch_d", "stoch_delta", "stoch_traj",
            "stoch_state", "atr", "atr_floor", "atr_state"]


def backend_name() -> str:
    return "ta" if (config.INDICATOR_BACKEND == "ta" and HAVE_TA) else "pandas"


def _rsi_js(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return out
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    ag, al = gains[:period].mean(), losses[:period].mean()
    val = lambda g, l: 100 - 100 / (1 + (100 if l == 0 else g / l))  # noqa: E731  (same quirk as the JS)
    out[period] = val(ag, al)
    for i in range(period + 1, len(close)):
        ag = (ag * (period - 1) + gains[i - 1]) / period
        al = (al * (period - 1) + losses[i - 1]) / period
        out[i] = val(ag, al)
    return out


def _stoch_pandas(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    rsi = pd.Series(_rsi_js(close.to_numpy(float), RSI_PERIOD), index=close.index)
    lo = rsi.rolling(STOCH_PERIOD).min()
    hi = rsi.rolling(STOCH_PERIOD).max()
    raw = ((rsi - lo) / (hi - lo) * 100).where(hi != lo, 50.0).where(rsi.rolling(STOCH_PERIOD).count() == STOCH_PERIOD)
    k = raw.rolling(3).mean()
    return k, k.rolling(3).mean()


def _stoch_ta(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    from ta.momentum import RSIIndicator
    raw = StochRSIIndicator(close=close, window=STOCH_PERIOD, smooth1=3, smooth2=3).stochrsi()  # 0..1
    # NaN has two causes: RSI not warmed up yet (keep NaN) or a completely flat RSI window, i.e. 0/0 (the app says 50).
    rsi_full = RSIIndicator(close=close, window=RSI_PERIOD).rsi().rolling(STOCH_PERIOD).count() == STOCH_PERIOD
    raw = raw.where(raw.notna() | ~rsi_full, 0.5) * 100
    k = raw.rolling(3).mean()
    return k, k.rolling(3).mean()


def indicator_frame(candles: list[dict], tf_ms: int = MIN_MS) -> pd.DataFrame:
    """Closed candles (oldest first) -> DataFrame with OHLCV + every indicator column for every row (NaN while warming up)."""
    df = pd.DataFrame(candles, columns=COLS) if candles else pd.DataFrame(columns=COLS)
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    n = len(df)
    nan = pd.Series(np.nan, index=df.index, dtype=float)
    if n == 0:
        for c in IND_COLS:
            df[c] = pd.Series(dtype=float)
        return df
    close, high, low = df["c"], df["h"], df["l"]
    use_ta = backend_name() == "ta"

    # --- Bollinger Bands (20, 2 sd, population sd) ---
    if n >= BB_PERIOD:
        if use_ta:
            bb = BollingerBands(close=close, window=BB_PERIOD, window_dev=2)
            df["bb_mid"], df["bb_upper"], df["bb_lower"] = bb.bollinger_mavg(), bb.bollinger_hband(), bb.bollinger_lband()
        else:
            mid = close.rolling(BB_PERIOD).mean()
            sd = close.rolling(BB_PERIOD).std(ddof=0)
            df["bb_mid"], df["bb_upper"], df["bb_lower"] = mid, mid + 2 * sd, mid - 2 * sd
    else:
        df["bb_mid"] = df["bb_upper"] = df["bb_lower"] = nan
    df["bb_state"] = np.where(df["bb_upper"].isna(), None,
                              np.where(close >= df["bb_upper"], "Upper_Stretched",
                                       np.where(close <= df["bb_lower"], "Lower_Stretched", "Mid_Band")))

    # --- Stochastic RSI (14/14, %K = 3-bar SMA). The app needs 3 valid %K values before it trusts the reading. ---
    if n > RSI_PERIOD + 1:
        k, d = (_stoch_ta if use_ta else _stoch_pandas)(close)
        k = k.where(k.notna().cumsum() >= 3)
        d = d.where(k.notna())
    else:
        k = d = nan
    df["stoch_k"], df["stoch_d"] = k, d
    delta = k - k.shift(1)
    df["stoch_delta"] = delta
    traj = np.where(k.isna(), None, np.where(delta > 0.5, "Rising", np.where(delta < -0.5, "Falling", "Flat")))
    df["stoch_traj"] = traj
    state = np.where(k.isna(), None, "Mixed")
    state = np.where((k > 60) & (traj == "Rising"), "High_Rising", state)
    state = np.where((k < 40) & (traj == "Falling"), "Low_Falling", state)
    state = np.where((k > 70) & (delta < 0), "Rollover_Top", state)
    state = np.where((k < 30) & (delta > 0), "Bounce_Bottom", state)
    df["stoch_state"] = state

    # --- ATR (period 15). true range excludes the first candle, exactly like the JS ---
    if n >= ATR_PERIOD + 1:
        if use_ta and config.ATR_SMOOTHING == "wilder":
            atr = AverageTrueRange(high=high, low=low, close=close, window=ATR_PERIOD).average_true_range()
        else:
            pc = close.shift(1)
            tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            tr.iloc[0] = np.nan
            atr = tr.rolling(ATR_PERIOD).mean()
    else:
        atr = nan
    df["atr"] = atr
    # floor = 7 bps of price per MINUTE, scaled by sqrt(candle/1m)
    df["atr_floor"] = close * ATR_FLOOR_BPS_PER_MIN * math.sqrt(tf_ms / MIN_MS)
    df["atr_state"] = np.where(atr.isna(), None, np.where(atr < df["atr_floor"], "Quiet/Asleep", "Active"))
    df.loc[atr.isna(), "atr_floor"] = np.nan
    return df


def latest(df: pd.DataFrame) -> dict:
    """Last row of an indicator_frame in the exact shape the JS engine keeps in `eng.ind`."""
    out = {"bb": None, "stoch": None, "atr": None}
    if df.empty:
        return out
    r = df.iloc[-1]
    if pd.notna(r["bb_mid"]):
        out["bb"] = dict(mid=float(r["bb_mid"]), upper=float(r["bb_upper"]), lower=float(r["bb_lower"]), state=r["bb_state"])
    if pd.notna(r["stoch_k"]):
        k, delta = float(r["stoch_k"]), float(r["stoch_delta"])
        out["stoch"] = dict(k=k, d=None if pd.isna(r["stoch_d"]) else float(r["stoch_d"]), prevK=k - delta, delta=delta,
                            traj=r["stoch_traj"], state=r["stoch_state"])
    if pd.notna(r["atr"]):
        out["atr"] = dict(atr=float(r["atr"]), state=r["atr_state"], floor=float(r["atr_floor"]))
    return out

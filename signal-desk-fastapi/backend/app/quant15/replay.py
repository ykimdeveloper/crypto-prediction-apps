"""Replay the ORIGINAL algorithm over stored 1m candles.

Same code path as live: indicators.indicator_frame -> Engine-shaped state -> strategy.compute_module_signals. One decision per
closed minute (live evaluates every second, so this is a slightly coarser view of the same rules). Entry / take-profit / settle
follow paper.py: first BUY per (strategy, window) with conf >= PAPER.minConf, TP when the modeled bid reaches entry+tp while more
than midManageUntil minutes remain, otherwise settle at 100/0. What is NOT replayed: the paper gates (they depend on live PnL history).

Honesty note: the up/down "book" is modeled by strategy.book_model (ATR-normal), not a real quote. Settlement outcomes (did the
window close above its open) are real; cents PnL is only as real as that modeled price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators
from ..constants import CONTRACT_MS, MIN_MS, PAPER, STRAT_KEYS, TIMING
from ..engine import Engine
from ..strategy import compute_module_signals

STEP_COLS = ["win_start", "m", "price", "strike", "atr", "p_book", "up"]


def to_candles(d1: pd.DataFrame) -> list[dict]:
    return [dict(t=int(t), o=float(o), h=float(h), l=float(l), c=float(c), v=float(v))
            for t, o, h, l, c, v in d1[["ts", "open", "high", "low", "close", "volume"]].itertuples(index=False, name=None)]


def _step(eng: Engine, frame: pd.DataFrame, i: int, cand: list[dict], win_start: int, strike: float, now: int) -> dict:
    eng.ind = indicators.latest(frame.iloc[: i + 1])
    eng.current, eng.contract_start, eng.contract_strike, eng.last_tick_at = cand[i], win_start, strike, int(now)
    return compute_module_signals(eng, MIN_MS, now)


def replay(d1: pd.DataFrame, min_conf: int | None = None) -> dict:
    """d1: 1m OHLCV frame (ts, open, high, low, close, volume) sorted by ts. Returns signals / windows / steps."""
    min_conf = PAPER["minConf"] if min_conf is None else min_conf
    d1 = d1.reset_index(drop=True)
    empty = dict(signals=[], windows=[], steps=pd.DataFrame(columns=STEP_COLS))
    if len(d1) < 60:
        return empty
    cand = to_candles(d1)
    frame = indicators.indicator_frame(cand)      # built once; indicators only look back <= ~30 candles (RSI seed washes out)
    ts = d1["ts"].to_numpy("int64")
    win = ts // CONTRACT_MS * CONTRACT_MS
    starts = np.flatnonzero(np.r_[True, win[1:] != win[:-1]])
    ends = np.r_[starts[1:], len(win)]
    eng = Engine(MIN_MS)
    signals, windows, steps = [], [], []
    for a, b in zip(starts, ends):
        W = int(win[a])
        if ts[a] != W or ts[b - 1] != W + CONTRACT_MS - MIN_MS:      # need the open AND the final minute of the window
            continue
        strike, close_end = cand[a]["o"], cand[b - 1]["c"]
        up = bool(close_end > strike)
        windows.append(dict(win_start=W, strike=strike, close_end=close_end, up=up))
        open_recs: dict[str, dict] = {}
        for i in range(a, b):
            now = int(ts[i]) + MIN_MS
            sig = _step(eng, frame, i, cand, W, strike, now)
            book = sig.get("book")
            if book is None:
                continue
            m = (now - W) // MIN_MS
            atr = (eng.ind["atr"] or {}).get("atr")
            steps.append((W, m, cand[i]["c"], strike, atr, book["pUp"] / 100.0, up))
            remaining = W + CONTRACT_MS - now
            for rec in open_recs.values():                             # take-profit first, like paper.apply
                if rec["exited"] or rec["tp"] is None or remaining <= TIMING["midManageUntil"] * MIN_MS:
                    continue
                bid = book["up"]["bid"] if rec["dir"] == "UP" else book["down"]["bid"]
                if bid >= rec["entry"] + rec["tp"]:
                    rec["exited"] = True
            for s in STRAT_KEYS:
                cur = sig[s]
                if s in open_recs or not cur["signal"].startswith("BUY") or (cur["conf"] or 0) < min_conf:
                    continue
                open_recs[s] = dict(strat=s, dir="UP" if cur["signal"].endswith("UP") else "DOWN", conf=cur["conf"], entry=cur["ask"],
                                    tp=cur.get("tp"), win_start=W, minute=int(m), strike=strike, story=sig["story"], lean=sig["lean"],
                                    p_hit=cur.get("pHit"), p_book_up=book["pUp"] / 100.0, exited=False)
        for rec in open_recs.values():
            won = close_end > strike if rec["dir"] == "UP" else close_end < strike
            settle = 100 - rec["entry"] if won else -rec["entry"]
            rec.update(win=bool(won), settle_pnl=float(settle), pnl=float(rec["tp"] if rec["exited"] else settle),
                       how="tp" if rec["exited"] else "settle", close_end=close_end)
            signals.append(rec)
    return dict(signals=signals, windows=windows, steps=pd.DataFrame(steps, columns=STEP_COLS))


def signals_now(d1_tail: pd.DataFrame) -> dict:
    """The original algorithm's current rev/mom/end signals from the newest closed 1m candle (used by /decide)."""
    d1_tail = d1_tail.reset_index(drop=True)
    cand = to_candles(d1_tail)
    frame = indicators.indicator_frame(cand)
    i = len(cand) - 1
    now = cand[i]["t"] + MIN_MS
    W = now // CONTRACT_MS * CONTRACT_MS
    if now % CONTRACT_MS == 0:                    # last minute of a window just closed: the next window hasn't started
        W = now - CONTRACT_MS
    first = next((c for c in cand if c["t"] == W), None)
    strike = first["o"] if first else cand[i]["c"]   # same fallback as Engine.strike_at
    return _step(Engine(MIN_MS), frame, i, cand, W, strike, now)

"""Port of src/lib/engine.js for one (exchange, symbol) 1-minute stream. Indicators come from app.indicators (ta)."""
from __future__ import annotations

import time

from .constants import CONTRACT_MS, MIN_MS, STRAT_KEYS
from .config import MAX_CANDLES
from . import indicators
from .strategy import compute_module_signals


class Engine:
    def __init__(self, tf_ms: int = MIN_MS):
        self.tf_ms = tf_ms
        self.candles: list[dict] = []      # closed candles only, oldest first
        self.current: dict | None = None   # forming candle (or the last closed one, same as the JS poll path)
        self.last_tick_at = 0
        self.contract_start: int | None = None
        self.contract_strike: float | None = None
        self.ind = {"bb": None, "stoch": None, "atr": None}
        self.frame = indicators.indicator_frame([], tf_ms)
        self.last_log = {k: None for k in STRAT_KEYS}
        self.signals: dict | None = None

    # --- data in ---
    def set_candles(self, closed: list[dict]):
        new = closed[-MAX_CANDLES:]
        if new and self.candles and len(new) == len(self.candles) and new[0]["t"] == self.candles[0]["t"] and new[-1] == self.candles[-1]:
            return False  # nothing closed or amended: keep the indicators we already have
        self.candles = new
        self.frame = indicators.indicator_frame(self.candles, self.tf_ms)
        self.ind = indicators.latest(self.frame)
        return True

    def update(self, closed: list[dict], forming: dict | None, now_ms: float | None = None) -> bool:
        now = int(time.time() * 1000) if now_ms is None else int(now_ms)
        changed = self.set_candles(closed)
        self.current = forming or (closed[-1] if closed else None)
        self.last_tick_at = now
        self.maybe_roll_contract(now)
        self.refresh_strike()
        return changed

    # --- contract clock (15 real minutes, aligned to :00/:15/:30/:45) ---
    def strike_at(self, win_start: int):
        if self.current and self.current["t"] == win_start:
            return self.current["o"]
        for c in reversed(self.candles[-1500:]):
            if c["t"] == win_start:
                return c["o"]
            if c["t"] < win_start:
                break
        return self.current["c"] if self.current else None

    def maybe_roll_contract(self, ts: int):
        win = (int(ts) // CONTRACT_MS) * CONTRACT_MS
        if self.contract_start != win:
            self.contract_start = win
            self.contract_strike = self.strike_at(win)

    def refresh_strike(self):
        if self.contract_start is None:
            return
        s = self.strike_at(self.contract_start)
        if s is not None:
            self.contract_strike = s

    # --- signals ---
    def evaluate(self, now_ms: float | None = None) -> dict:
        sig = compute_module_signals(self, self.tf_ms, now_ms)
        self.signals = sig
        return sig


def live_price(eng: Engine):
    if eng.current:
        return eng.current["c"]
    if eng.candles:
        return eng.candles[-1]["c"]
    return None

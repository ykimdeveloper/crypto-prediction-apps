"""One definition of how a 15m up/down window settles. Import this instead of writing `>` or `>=` inline.

Contract rule (Coinbase BTC/ETH 15 min, CF Benchmarks BRTI as the source): the window resolves UP ("Yes") when the
60-second average of the index before the window END is AT LEAST the 60-second average before the window START.
So an exact tie settles UP, and a DOWN position wins only on a strictly lower end value.

The strike is the START reference, the end value is the END reference. Neither is the exchange's last trade or a
candle's open/close; candle values are an approximation of the index averages and should be labeled that way.
"""
from __future__ import annotations

TIE_SETTLES = "UP"


def settles_up(end_ref: float, strike: float) -> bool:
    return end_ref >= strike


def settle_dir(end_ref: float, strike: float) -> str:
    return "UP" if settles_up(end_ref, strike) else "DOWN"


def position_wins(direction: str, end_ref: float, strike: float) -> bool:
    up = settles_up(end_ref, strike)
    return up if direction == "UP" else not up

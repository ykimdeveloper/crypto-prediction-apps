"""1:1 port of src/lib/strategy.js (lean/story classifier, modeled Up/Down book, the three strategies)."""
from __future__ import annotations

import math
import time

from .constants import BOOK, CONTRACT_MS, EOM_CONF_SIGMA, EOM_INDICATOR_WEIGHT, MID, MIN_MS, PRICE, TIMING
from . import config


def js_round(x: float) -> int:
    """Math.round: halves go up (Python's round() is banker's rounding)."""
    return math.floor(x + 0.5)


def num(x):
    """How JS prints a number in a template string: 42.0 -> '42'."""
    return str(int(x)) if isinstance(x, float) and x == int(x) else str(x)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def round1(v):
    return js_round(v * 10) / 10


def round3(v):
    return js_round(v * 1000) / 1000


def f2(v):
    """Mirror of the JS f2: pre-round so toFixed(2) and Python's formatter can't disagree on exact .xx5 ties."""
    return f"{js_round(v * 100) / 100:.2f}"


def fmt(price):
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return "—"
    d = 8 if price < 0.01 else 6 if price < 0.1 else 4 if price < 5 else 3 if price < 100 else 2
    return f"{price:.{d}f}"


def clamp_conf(v):
    return max(50, min(97, js_round(v)))


def _erf(x):  # same Abramowitz-Stegun approximation as the JS so both sides agree to the last digit
    sign = -1 if x < 0 else 1
    ax = abs(x)
    t = 1 / (1 + 0.3275911 * ax)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-ax * ax)
    return sign * y


def normal_cdf(z):
    return 0.5 * (1 + _erf(z / math.sqrt(2)))


def bb_position(bb, price):
    if not bb:
        return 0
    up = (bb["upper"] - bb["mid"]) or 1e-9
    dn = (bb["mid"] - bb["lower"]) or 1e-9
    return (price - bb["mid"]) / up if price >= bb["mid"] else (price - bb["mid"]) / dn


def evaluate_lean_and_story(ind, price):
    if not ind["bb"] or not ind["stoch"]:
        return dict(story="Mixed / None", lean=0, momentum=0, snap=0)
    k, traj = ind["stoch"]["k"], ind["stoch"]["traj"]
    pos = bb_position(ind["bb"], price)

    momentum = 0
    if traj == "Rising" and k >= 45:
        momentum = 0.35 + (0.65 * (k - 45)) / 55
    elif traj == "Falling" and k <= 55:
        momentum = -(0.35 + (0.65 * (55 - k)) / 55)

    breakout = (pos >= 1 and momentum > 0) or (pos <= -1 and momentum < 0)
    if breakout:
        momentum = clamp(momentum * 1.25, -1, 1)

    snap_turn = 0
    if traj == "Rising" and k < 30:
        turn = (30 - k) / 30
        near = clamp((-pos - 0.2) / 0.8, 0, 1)
        snap_turn = (0.4 + 0.6 * turn) * (0.5 + 0.5 * near)
    elif traj == "Falling" and k > 70:
        turn = (k - 70) / 30
        near = clamp((pos - 0.2) / 0.8, 0, 1)
        snap_turn = -(0.4 + 0.6 * turn) * (0.5 + 0.5 * near)
    snap_edge = 0
    if pos <= -0.85:
        snap_edge = 0.35 * clamp(0.5 + (-pos - 0.85) / 0.3, 0.5, 1)
    elif pos >= 0.85:
        snap_edge = -0.35 * clamp(0.5 + (pos - 0.85) / 0.3, 0.5, 1)
    snap = clamp(snap_turn + snap_edge, -1, 1)

    sign = lambda v: (v > 0) - (v < 0)  # noqa: E731
    both_weak = abs(momentum) < 0.15 and abs(snap) < 0.15
    fighting = (not both_weak) and abs(momentum) >= 0.15 and abs(snap) >= 0.15 and sign(momentum) != sign(snap)

    if breakout:
        story, lean = "Momentum Continuation", momentum
    elif both_weak:
        story, lean = "Mixed / None", ((momentum + snap) / 2) * 0.4
    elif fighting:
        story, lean = "Mixed / None", ((momentum + snap) / 2) * 0.5
    elif abs(snap) > abs(momentum):
        story, lean = "Mean Reversion", snap
    else:
        story, lean = "Momentum Continuation", momentum
    return dict(story=story, lean=clamp(lean, -1, 1), momentum=momentum, snap=snap)


def book_model(eng, price, remaining_ms, tf_ms):
    if not eng.ind["atr"] or eng.contract_strike is None or price is None:
        return None
    candles_left = max(remaining_ms / tf_ms, 0.02)
    sigma = (eng.ind["atr"]["atr"] or 1e-12) * math.sqrt(candles_left) * BOOK["sigmaFudge"]
    p_up = clamp(normal_cdf((price - eng.contract_strike) / sigma) * 100, 0.5, 99.5)
    h = BOOK["halfSpread"]
    up_bid = round1(clamp(p_up - h, 0.1, 99.8))
    up_ask = round1(clamp(p_up + h, 0.2, 99.9))
    return dict(pUp=p_up, up=dict(bid=up_bid, ask=up_ask), down=dict(bid=round1(100 - up_ask), ask=round1(100 - up_bid)))


def _short(story):
    return "revert" if story == "Mean Reversion" else "mom" if story == "Momentum Continuation" else "mixed"


def WAIT(reason, conf=None):
    return dict(signal="WAIT", conf=conf, reason=reason)


def compute_module_signals(eng, tf_ms, now_ms=None):
    from .engine import live_price
    now = time.time() * 1000 if now_ms is None else now_ms
    price = live_price(eng)
    warming = not eng.ind["bb"] or not eng.ind["stoch"] or not eng.ind["atr"] or eng.contract_start is None or price is None
    stale = bool(eng.last_tick_at) and now - eng.last_tick_at > config.STALE_MS
    if warming or stale:
        why = "Feed stale for this window." if stale else "Warming up — not enough closed candles yet."
        return dict(module="—", story=None, lean=0, book=None, remMin=None, rev=WAIT(why), mom=WAIT(why), end=WAIT(why))

    ls = evaluate_lean_and_story(eng.ind, price)
    story, lean = ls["story"], ls["lean"]
    remaining_ms = eng.contract_start + CONTRACT_MS - now
    rem_min = remaining_ms / MIN_MS
    strike = eng.contract_strike
    book = book_model(eng, price, remaining_ms, tf_ms)
    sleep_ok = eng.ind["atr"]["state"] != "Quiet/Asleep"
    d = "UP" if lean > 0 else "DOWN" if lean < 0 else None
    side_ask = lambda x: book["up"]["ask"] if x == "UP" else book["down"]["ask"]  # noqa: E731
    # one "sigma" of the path still to come, on the same scale the modeled book uses
    candles_left = max(remaining_ms / tf_ms, 0.02)
    path_scale = (eng.ind["atr"]["atr"] * math.sqrt(candles_left)) or 1e-12

    def mid_fork(kind):
        if kind == "rev" and not MID["revEnabled"]:
            return WAIT("Mean reversion is off in v0.2 — mid-market only takes the momentum story.")
        want = "Mean Reversion" if kind == "rev" else "Momentum Continuation"
        if story != want:
            return WAIT(f"Not {'rev' if kind == 'rev' else 'mom'} — story is {_short(story)}.")
        if rem_min < TIMING["midEntryMinRemaining"]:
            return WAIT(f"Under {TIMING['midEntryMinRemaining']}:00 left.")
        if not sleep_ok:
            return WAIT("ATR asleep — sleep filter blocks the mid-market forks.")
        if not d:
            return WAIT("No lean.")
        k, traj, state = eng.ind["stoch"]["k"], eng.ind["stoch"]["traj"], eng.ind["stoch"]["state"]
        conf = clamp_conf(50 + abs(lean) * 38 + 8)

        # momentum only: price must already be a little on the lean's side of the strike, but not run out
        side_z = None
        if kind == "mom":
            side_z = ((price - strike) * (1 if d == "UP" else -1)) / path_scale
            if side_z < MID["sideMinZ"]:
                return WAIT(f"Price isn't on the {d} side of the strike yet ({f2(side_z)}σ, need {f2(MID['sideMinZ'])}σ).", conf)
            if side_z > MID["sideMaxZ"]:
                return WAIT(f"Move already ran: {f2(side_z)}σ past the strike (max {f2(MID['sideMaxZ'])}σ).", conf)

        if kind == "rev":
            stoch_ok = (state == "Bounce_Bottom" or (k <= 40 and traj == "Rising")) if d == "UP" else (state == "Rollover_Top" or (k >= 60 and traj == "Falling"))
        else:
            stoch_ok = (50 <= k <= 85 and traj == "Rising") if d == "UP" else (15 <= k <= 50 and traj == "Falling")
        if not stoch_ok:
            return WAIT("Stoch not confirmed.", conf)
        ask = side_ask(d)
        lo, hi = PRICE["mmrBand"] if kind == "rev" else PRICE["mmcBand"]
        if ask < PRICE["minEntry"]:
            return WAIT(f"{d} is {num(ask)}¢ — under the {PRICE['minEntry']}¢ minimum.", conf)
        if ask < lo:
            return WAIT(f"{d} is {num(ask)}¢ — below the {lo}–{hi}¢ entry band.", conf)
        if ask > hi:
            return WAIT(f"{d} is {num(ask)}¢ — above the {lo}–{hi}¢ entry band.", conf)
        tp = PRICE["tpMmr"] if kind == "rev" else PRICE["tpMmc"]

        # chance of hitting the take-profit: fair value is a martingale ending at 0/100, so P(reach L from p) = p / L, with
        # p = our probability this side finishes in the money and L = ask + tp + half-spread (where the bid sits tp above the ask)
        p_hit = None
        if kind == "mom":
            side_prob = normal_cdf(side_z / MID["hitSigmaFudge"]) * 100
            p_hit = round3(min(1, side_prob / (ask + tp + BOOK["halfSpread"])))
            if p_hit < MID["minHitProb"]:
                return WAIT(f"+{tp}¢ only ~{js_round(p_hit * 100)}% likely from {num(ask)}¢ (need {js_round(MID['minHitProb'] * 100)}%).", conf)
        reason = (f"Snapback: {d} at {num(ask)}¢, target +{tp}¢, no stop." if kind == "rev"
                  else f"Breakout running: {d} at {num(ask)}¢, target +{tp}¢ (~{js_round(p_hit * 100)}% to hit), no stop.")
        return dict(signal=f"BUY {d}", conf=conf, ask=ask, tp=tp, pHit=p_hit, reason=reason)

    rev, mom = mid_fork("rev"), mid_fork("mom")

    if rem_min < TIMING["eomMinRemaining"]:
        end = WAIT(f"Closed under {TIMING['eomMinRemaining']}:00 remaining.")
    else:
        price_up = price >= strike
        e_dir = "UP" if price_up else "DOWN"
        ask = side_ask(e_dir)
        cushion = abs(price - strike)
        buffer = path_scale
        # indicators are only a light early nudge: full weight at 8:00, fading linearly to 0 at 2:00
        fade = clamp((rem_min - TIMING["eomMinRemaining"]) / (TIMING["eomMaxRemaining"] - TIMING["eomMinRemaining"]), 0, 1)
        ind_w = EOM_INDICATOR_WEIGHT * fade
        stat_conf = normal_cdf(cushion / ((buffer * EOM_CONF_SIGMA) or 1e-12)) * 100
        agree = lean if e_dir == "UP" else -lean
        conf = clamp(js_round((1 - ind_w) * stat_conf + ind_w * (50 + 50 * agree)), 1, 99)
        lo, hi = PRICE["eomBand"]
        if rem_min > TIMING["eomMaxRemaining"]:
            end = WAIT(f"Leaning {e_dir} ({num(ask)}¢) — buys open at {TIMING['eomMaxRemaining']}:00 remaining.", conf)
        elif ask > hi:
            end = dict(signal=f"HOLD {e_dir}", conf=conf, reason=f"above {hi}¢ — nothing left to earn.")
        elif ask < lo:
            end = WAIT(f"{e_dir} is {num(ask)}¢ — below {lo}¢, cushion {fmt(cushion)} vs buffer {fmt(buffer)}.", conf)
        else:
            end = dict(signal=f"BUY {e_dir}", conf=conf, ask=ask, tp=None,
                       reason=f"{lo}–{hi}¢ · cushion {fmt(cushion)} vs buffer {fmt(buffer)}. Holds to settlement.")

    overlap = TIMING["midEntryMinRemaining"] < rem_min <= TIMING["eomMaxRemaining"]
    module = "MID" if rem_min > TIMING["eomMaxRemaining"] else "MID+END" if overlap else "END" if rem_min >= TIMING["eomMinRemaining"] else "CLOSING"
    return dict(module=module, story=story, lean=lean, book=book, remMin=rem_min, rev=rev, mom=mom, end=end)

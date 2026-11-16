"""Rule-based engine: the chart-reading logic Claude applied in chat, made explicit and testable.

Indicator frames (`inds`) are indexed by bar CLOSE time so lookups never see the future.
"""
import math, os

import numpy as np
import pandas as pd

H = 15  # horizon, minutes
K_DRIFT = float(os.getenv("K_DRIFT", "0.15"))      # expected drift (in sigmas) per 1.0 of composite score
BUY_TH = float(os.getenv("BUY_THRESHOLD", "0.2"))  # composite needed for BUY / AVOID
W = {"1m": .10, "5m": .25, "10m": .15, "15m": .25, "30m": .15, "1h": .10}  # 5m/15m set direction, 1m timing, 1h regime


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def sigma_series(close):
    return np.log(close).diff().ewm(span=60).std()


def sigma_1m(close):
    v = sigma_series(close).iloc[-1]
    return 0.0 if np.isnan(v) else float(v)


def tf_score(x, p):
    """Score one timeframe in [-1, 1] (+ bullish) from its last closed bar `x` and the bar before `p`."""
    parts, notes = [], []

    def add(v, note):
        parts.append(v)
        if abs(v) >= 0.4:
            notes.append(note)

    if not np.isnan(x.vstop):
        up = x.close > x.vstop
        add(0.5 if up else -0.5, "above VolStop" if up else "below VolStop")
    if not np.isnan(x.macd_h):
        rising = x.macd_h > p.macd_h
        add(0.4 * np.sign(x.macd_h) + (0.2 if rising else -0.2), f"MACD hist {'+' if x.macd_h > 0 else '-'} {'rising' if rising else 'falling'}")
    if not np.isnan(x.pctb):  # band touch = bounce/rejection; otherwise mid-band side
        v = 0.6 if x.pctb < 0.1 else -0.6 if x.pctb > 0.9 else 0.3 * np.sign(x.pctb - 0.5)
        add(v, f"%B {x.pctb:.2f} at {'lower' if x.pctb < 0.1 else 'upper'} band")
    if not np.isnan(x.k):  # extremes only count when K/D turns; otherwise weak
        if x.k < 20:
            v = 0.6 if x.k > x.d else 0.2
        elif x.k > 80:
            v = -0.6 if x.k < x.d else -0.2
        else:
            v = 0.2 if x.k > x.d else -0.2
        add(v, f"StochRSI K{x.k:.0f}/D{x.d:.0f} turning {'up from oversold' if v > 0 else 'down from overbought'}")
    return (float(np.clip(np.mean(parts) * 1.5, -1, 1)) if parts else 0.0), notes


def decide(inds, t, px, sigma, imb=0.0, threshold=None, k=K_DRIFT):
    sc, reasons, atr = {}, [], {}
    for n, ind in inds.items():
        i = ind.index.searchsorted(t, side="right") - 1  # last bar closed at or before t
        if i < 1:
            return None
        x, p = ind.iloc[i], ind.iloc[i - 1]
        sc[n], notes = tf_score(x, p)
        atr[n] = float(x.atr)
        reasons += [f"{n}: {s}" for s in notes]
    comp = sum(W[n] * s for n, s in sc.items()) + 0.15 * imb
    bull, bear = sum(s > .1 for s in sc.values()), sum(s < -.1 for s in sc.values())
    if comp >= BUY_TH and bull >= 4 and sc["5m"] > -.1 and sc["15m"] > -.1:
        act = "BUY"
    elif comp <= -BUY_TH and bear >= 4:
        act = "AVOID"
    else:
        act = "WAIT"
    sh = sigma * math.sqrt(H)
    mu = k * comp * sh
    out = {"action": act, "composite": round(comp, 3), "strength": int(min(99, abs(comp) * 200)),
           "scores": {n: round(s, 2) for n, s in sc.items()}, "bull_tfs": int(bull), "bear_tfs": int(bear),
           "reasons": reasons[:8], "sigma_15m_pct": round(sh * 100, 3),
           "range_1sd": [round(px * math.exp(mu - sh), 2), round(px * math.exp(mu + sh), 2)],
           "p_up": round(phi(mu / sh), 3) if sh > 0 else 0.5}
    a5 = atr["5m"]
    if act == "BUY" and a5 == a5:  # stop 1.25 x ATR5m, targets at 1.5R and 2.5R
        out.update(entry=round(px, 2), stop=round(px - 1.25 * a5, 2),
                   target_1=round(px + 1.875 * a5, 2), target_2=round(px + 3.125 * a5, 2))
    if threshold and sh > 0:
        d = math.log(px / threshold)  # >0: price above threshold
        out["threshold"] = threshold
        out["p_above"] = round(phi((d + mu) / sh), 3)
        z = (abs(d) - (mu if threshold > px else -mu)) / sh  # reflection-principle approximation
        out["p_touch"] = round(min(1.0, 2 * (1 - phi(z))), 3) if z > 0 else 1.0
    return out


def context_text(px, sigma, atr, threshold=None):
    """Pure arithmetic handed to Claude (no verdict, so the two approaches stay independent)."""
    sh = sigma * math.sqrt(H)
    L = ["=== COMPUTED CONTEXT (arithmetic only) ===",
         f"price {px:,.2f} | realized 1m vol {sigma * 100:.4f}% | 15m 1-sigma move {px * sh:,.2f} ({sh * 100:.3f}%) | "
         f"1-sigma range {px * math.exp(-sh):,.2f} .. {px * math.exp(sh):,.2f}",
         "ATR by timeframe: " + ", ".join(f"{n} {v:,.2f}" for n, v in atr.items())]
    if threshold and sh > 0:
        gap = threshold - px
        L.append(f"threshold {threshold:,.2f}: {gap:+,.2f} away = {gap / (px * sh):+.2f} sigma(15m) = {gap / atr['1m']:+.2f} x 1m ATR; "
                 f"driftless P(above at +15m) = {phi(math.log(px / threshold) / sh):.0%}")
    return "\n".join(L)


def backtest(inds, close1m, k=K_DRIFT, step=15, warmup_h=40):
    """Replay decide() at every 15-min boundary and compare with what price actually did over the next 15 min.
    Depth is not available historically, so imbalance is 0 here."""
    sig = sigma_series(close1m)
    start = close1m.index[0] + pd.Timedelta(hours=warmup_h)
    rows, one = [], pd.Timedelta(minutes=1)
    for t in close1m.index:
        if t < start or t.minute % step:
            continue
        try:
            p0, p1, s = close1m.at[t - one], close1m.at[t + (H - 1) * one], sig.at[t - one]
        except KeyError:
            continue
        if not s > 0:
            continue
        d = decide(inds, t, p0, s, 0.0, None, k)
        if d:
            rows.append((d["action"], math.log(p1 / p0), s * math.sqrt(H), d["p_up"]))
    if not rows:
        return {"n": 0, "note": "not enough history; raise LOOKBACK_MINUTES"}
    df = pd.DataFrame(rows, columns=["a", "r", "sh", "p"])
    df["up"] = (df.r > 0).astype(float)
    by = {a: {"n": int(len(x)), "up_pct": round(x.up.mean() * 100, 1), "mean_ret_bps": round(x.r.mean() * 1e4, 2)}
          for a, x in df.groupby("a")}
    return {"n": int(len(df)), "by_action": by, "base_up_pct": round(df.up.mean() * 100, 1),
            "within_1sd_pct": round(((df.r / df.sh).abs() < 1).mean() * 100, 1),  # ~68 if the vol model is calibrated
            "brier_model": round(((df.p - df.up) ** 2).mean(), 4), "brier_coinflip": round(((0.5 - df.up) ** 2).mean(), 4),
            "note": "Model beats coin-flip only if brier_model < brier_coinflip. Small samples are noisy."}

"""Clock-aligned 15-min Up/Down contract engine. Pure logic (no I/O, no LLM).

Strike = price the window opened at. Modeled book: P(up) = Phi(dist / (ATR * sqrt(candles_left) * 1.4)),
quoted in cents with a 1c spread. Three strategies, paper trading, strategy-wide gates, exit alerts.
"""
import math, os
from collections import deque

WIN, FIT = 900, 1.4
ASLEEP = float(os.getenv("ASLEEP_RATIO", "0.75"))  # ATR below this fraction of its 50-bar average = "asleep"
NAMES = {"MR": "Mid-market Mean Reversion", "MOM": "Mid-market Momentum", "END": "End-of-market"}
TARGET = {"MR": 15, "MOM": 20}  # take-profit, cents above entry


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def model_book(spot, strike, atr, secs_left):
    den = atr * math.sqrt(max(secs_left, 1) / 60) * FIT
    p = phi((spot - strike) / den) if den > 0 else (1.0 if spot > strike else 0.0 if spot < strike else 0.5)
    mid = p * 100
    ub, ua = max(0.1, mid - 0.5), min(99.9, mid + 0.5)
    return {"p_up": p, "up_bid": ub, "up_ask": ua, "down_bid": max(0.1, 100 - ua), "down_ask": min(99.9, 100 - ub)}


class Engine:
    def __init__(self):
        self.cfg = {"fee": 0.8, "stop_pct": 75, "end_tp": 99, "min_conf": 80, "mid_on": True, "mr_on": True}
        self.reset()

    def reset(self):
        self.pos = {k: None for k in NAMES}
        self.fired = {k: None for k in NAMES}
        self.gate = {k: "open" for k in NAMES}
        self.trades, self.log = [], deque(maxlen=200)
        self.seq, self.wid, self.last, self.view = 0, None, None, None

    # ---- helpers
    def _ev(self, ts, kind, k, text):
        self.seq += 1
        self.log.appendleft({"id": self.seq, "ts": ts, "kind": kind, "strategy": k, "text": text,
                             "alert": self.gate[k] == "open"})

    def _close(self, k, ts, exit_px, reason):
        p = self.pos[k]
        net = exit_px - p["entry"] - p["fee"]  # TP = resting limit and settlement are free; only entry fee
        self.trades.append({"strategy": k, "side": p["side"], "entry": p["entry"], "exit": exit_px, "net": net,
                            "reason": reason, "closed": ts})
        self.pos[k] = None
        self._ev(ts, "TAKE PROFIT" if reason == "tp" else "SETTLED", k,
                 f"{p['side']} {p['entry']:.1f}->{exit_px:.0f}¢ net {net:+.1f}¢")

    def _avg(self, k, ts, secs):
        xs = [t["net"] for t in self.trades if t["strategy"] == k and t["closed"] >= ts - secs]
        return (sum(xs) / len(xs) if xs else None)

    def _gates(self, ts):
        for k in NAMES:
            if k == "END":
                h, r = self._avg(k, ts, 7200), self._avg(k, ts, 3600)
                hold, rel = h is not None and h <= -2, r is not None and r > 0
            else:
                h, r = self._avg(k, ts, 3600), self._avg(k, ts, 1800)
                hold, rel = h is not None and h <= -2, r is not None and r >= 2.5
            if self.gate[k] == "open" and hold:
                self.gate[k] = "held"
                self._ev(ts, "GATE", k, "held: trailing average <= -2¢")
                self.log[0]["alert"] = False
            elif self.gate[k] == "held" and rel:
                self.gate[k] = "open"
                self._ev(ts, "GATE", k, "released")

    # ---- main step
    def tick(self, ts, spot, strike, f):
        """f: k, k_prev, pctb, atr, atr_avg (1m indicators), lean (composite), strength (0-99)."""
        c, wid = self.cfg, int(ts // WIN)
        secs = (wid + 1) * WIN - ts
        if self.wid is not None and wid != self.wid and self.last:  # new window: settle the old one
            ls, lk = self.last
            for k, p in list(self.pos.items()):
                if p:
                    self._close(k, ts, 100 if (ls > lk) == (p["side"] == "UP") else 0, "settle")
        self.wid, self.last = wid, (spot, strike)
        b = model_book(spot, strike, f["atr"], secs)
        asleep = f["atr"] < ASLEEP * f["atr_avg"]
        k_, kp, pb = f["k"], f["k_prev"], f["pctb"]
        stretched = (k_ >= 80 and pb >= .85) or (k_ <= 20 and pb <= .15)
        story = "Mean Reversion" if stretched else "Momentum" if ((k_ >= 40 and k_ > kp) or (k_ <= 60 and k_ < kp)) else "Range"
        ask = lambda s: b["up_ask"] if s == "UP" else b["down_ask"]
        bid = lambda s: b["up_bid"] if s == "UP" else b["down_bid"]
        conf = lambda s: f["strength"] if f["lean"] != 0 and (f["lean"] > 0) == (s == "UP") else 0

        def mid(key, side, label, lo, hi):
            r = {"action": "WAIT", "side": side, "conf": 0, "note": f"Not {label} — story is {story.lower()}."}
            if not c["mid_on"] or (key == "MR" and not c["mr_on"]):
                return {**r, "action": "OFF", "note": "Strategy switched off."}
            if side is None:
                return r
            a, cf = ask(side), conf(side)
            r.update(conf=cf, ask=round(a, 1))
            if secs < 300: r["note"] = "Under 5:00 left."
            elif asleep: r["note"] = "Market asleep."
            elif not lo <= a <= hi: r["note"] = f"Ask {a:.0f}¢ outside {lo}-{hi}¢."
            elif cf < c["min_conf"]: r["note"] = f"Confidence {cf}% < {c['min_conf']}%."
            else: r.update(action=f"BUY {side}", note=f"Buy {side} at {a:.1f}¢, target +{TARGET[key]}¢.")
            return r

        def end():
            s = "UP" if spot >= strike else "DOWN"
            cf, a = round((b["p_up"] if s == "UP" else 1 - b["p_up"]) * 100), ask(s)
            r = {"action": "WAIT", "side": s, "conf": cf, "ask": round(a, 1), "note": "Opens at 8:00 left."}
            if 120 <= secs <= 480:
                if cf < c["min_conf"]: r["note"] = f"Confidence {cf}% < {c['min_conf']}%."
                elif a > 98: r.update(action=f"HOLD {s}", note="Ask above 98¢ — nothing to buy.")
                elif a >= 85: r.update(action=f"BUY {s}", note=f"Buy {s} at {a:.1f}¢, rides to settlement.")
                else: r["note"] = f"Ask {a:.0f}¢ below 85¢."
            elif secs < 120:
                r["note"] = "Window closed (under 2:00)."
            return r

        up_rev = k_ <= 20 and k_ > kp and pb <= .15
        dn_rev = k_ >= 80 and k_ < kp and pb >= .85
        mom = None if stretched else "UP" if (k_ >= 40 and k_ > kp) else "DOWN" if (k_ <= 60 and k_ < kp) else None
        cand = {"MR": lambda: mid("MR", "UP" if up_rev else "DOWN" if dn_rev else None, "revert", 35, 50),
                "MOM": lambda: mid("MOM", mom, "mom", 50, 78), "END": end}
        recs = {}
        for key in NAMES:
            p = self.pos[key]
            if p:  # manage open position
                bd = bid(p["side"])
                if key in TARGET and secs > 60 and bd >= p["entry"] + TARGET[key]:
                    self._close(key, ts, p["entry"] + TARGET[key], "tp")
                else:
                    if bd <= p["entry"] * c["stop_pct"] / 100 and "stop" not in p["alerted"]:
                        p["alerted"].append("stop")
                        self._ev(ts, "STOP", key, f"bid {bd:.1f}¢ <= {c['stop_pct']}% of entry — press SELL")
                    if key == "END" and bd >= c["end_tp"] and "tp" not in p["alerted"]:
                        p["alerted"].append("tp")
                        self._ev(ts, "TAKE PROFIT", key, f"bid {bd:.1f}¢ >= {c['end_tp']}¢ — press SELL")
            p = self.pos[key]
            if p:
                rec = {"action": f"HOLD {p['side']}", "side": p["side"], "conf": p["conf"],
                       "note": f"In {p['side']} at {p['entry']:.1f}¢. " + ("No stop; settles at window close." if key == "END" else f"Target +{TARGET[key]}¢.")}
            elif self.fired[key] == wid:
                rec = {"action": "DONE", "conf": 0, "note": "Fired once this window."}
            else:
                rec = cand[key]()
                if rec["action"].startswith("BUY"):
                    self.pos[key] = {"side": rec["side"], "entry": rec["ask"], "fee": c["fee"], "conf": rec["conf"], "alerted": []}
                    self.fired[key] = wid
                    self._ev(ts, "BUY", key, f"{rec['action']} · {rec['conf']}% @ {rec['ask']}¢")
            rec["gate"] = self.gate[key]
            recs[key] = rec
        self._gates(ts)
        r1 = lambda v: round(v, 1)
        self.view = {"ts": ts, "wid": wid, "secs_left": round(secs), "spot": spot, "strike": strike, "dist": spot - strike,
                     "book": {kk: (round(v, 3) if kk == "p_up" else r1(v)) for kk, v in b.items()},
                     "feat": {"k": r1(k_), "bb": "Upper Stretched" if pb >= .85 else "Lower Stretched" if pb <= .15 else "Mid",
                              "atr": round(f["atr"], 3), "asleep": bool(asleep), "lean": round(f["lean"], 2),
                              "strength": f["strength"], "story": story}, "recs": recs}

    def stats(self, ts):
        out = {}
        for k in NAMES:
            row = {}
            for label, secs in (("ALL", None), ("24H", 86400), ("2H", 7200)):
                xs = [t["net"] for t in self.trades if t["strategy"] == k and (secs is None or t["closed"] >= ts - secs)]
                row[label] = {"n": len(xs), "w": sum(x > 0 for x in xs), "avg": round(sum(xs) / len(xs), 1) if xs else None,
                              "pnl": round(sum(xs) / 100, 2)}
            out[k] = row
        return out

    def snapshot(self, ts):
        if not self.view:
            return {"ready": False}
        return {**self.view, "ready": True, "stats": self.stats(ts), "gates": dict(self.gate), "cfg": self.cfg,
                "log": list(self.log)[:30], "age": round(ts - self.view["ts"], 1)}

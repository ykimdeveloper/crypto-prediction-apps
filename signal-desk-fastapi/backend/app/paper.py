"""Port of src/lib/paper.js: paper trades, strategy gates, take-profit / stop optimizer and the stats table the PaperPanel draws.

Records are kept GROSS of fees (the fee setting re-prices history at read time), exactly like the browser version.
Every state change is appended to `paper.journal` as (event, record) so the hub can write it to SQLite:
  ("open", rec) when a position is opened, ("close", rec) when it takes profit or settles.
"""
from __future__ import annotations

from .constants import CONTRACT_MS, MIN_MS, PAPER, STRAT_KEYS, TIMING
from .engine import live_price

H = 3_600_000


def _gate_cfg(s):
    return PAPER["gates"]["end"] if s == "end" else PAPER["gates"]["mid"]


class Paper:
    def __init__(self, done=None):
        self.open: dict[str, dict] = {}
        self.done: list[dict] = list(done or [])
        self.gate: dict[str, dict] = {}
        self.rev = 0
        self.fee_cents = 0.0
        self.disabled: set[str] = set()
        self.exit_cfg: dict | None = None      # {"stopPct": 0..1, "tpEnd": cents}
        self.journal: list[tuple[str, dict]] = []
        self._opt_key = ""
        self._opt_val: dict = {}

    def configure(self, fee_cents=None, mid_enabled=None, exit_cfg=None):
        if fee_cents is not None:
            self.fee_cents = float(fee_cents)
        if mid_enabled is not None:
            self.disabled = set() if mid_enabled else {"rev", "mom"}
        if exit_cfg is not None:
            self.exit_cfg = exit_cfg or None


# ---------------- stats ----------------
def tally(recs):
    n = len(recs)
    w = sum(1 for r in recs if r["pnl"] > 0)
    total = sum(r["pnl"] for r in recs)
    srt = sorted(recs, key=lambda r: r["end"])
    streak, last = 0, None
    if n:
        last = srt[-1]["pnl"] > 0
        for r in reversed(srt):
            if (r["pnl"] > 0) != last:
                break
            streak += 1
    return dict(n=n, w=w, pct=(w / n if n else None), sum=total, avg=(total / n if n else None),
                streak=(f"{streak}{'W' if last else 'L'}" if n else "—"))


def _within(recs, now, span):
    return [r for r in recs if now - r["end"] <= span]


def _net(r, fee):
    if not fee:
        return r
    return {**r, "pnl": r["pnl"] - fee, "settlePnl": None if r.get("settlePnl") is None else r["settlePnl"] - fee}


def _net_all(recs, fee):
    return [_net(r, fee) for r in recs] if fee else recs


def _span_label(minutes):
    return f"{minutes // 60}h" if minutes >= 60 else f"{minutes}m"


def update_gates(paper: Paper, tf, now):
    for s in STRAT_KEYS:
        cfg = _gate_cfg(s)
        recs = _net_all([r for r in paper.done if r["tf"] == tf and r["strat"] == s], paper.fee_cents)
        st = paper.gate.setdefault(f"{tf}|{s}", {"held": False})
        hold = tally(_within(recs, now, cfg["holdSpanMin"] * 60000))
        rel = tally(_within(recs, now, cfg["releaseSpanMin"] * 60000))
        if not st["held"]:
            if hold["n"] >= PAPER["gateMinN"] and hold["avg"] <= cfg["holdBelow"]:
                st["held"] = True
        elif rel["n"] >= 1 and rel["avg"] >= cfg["releaseAtLeast"]:
            st["held"] = False
        st["hold"] = dict(label=_span_label(cfg["holdSpanMin"]), n=hold["n"], avg=hold["avg"])
        st["rel"] = dict(label=_span_label(cfg["releaseSpanMin"]), n=rel["n"], avg=rel["avg"])
        st["cfg"] = cfg


def gate_held(paper, tf, s):
    g = paper.gate.get(f"{tf}|{s}")
    return bool(g and g["held"])


def _cents(c):
    return "—" if c is None else f"{'+' if c >= 0 else '−'}{abs(c):.1f}¢"


def _gate_reason(paper, tf, s):
    g = paper.gate[f"{tf}|{s}"]
    return f"Gate closed: {g['hold']['label']} avg {_cents(g['hold']['avg'])} (n={g['hold']['n']}); reopens when {g['rel']['label']} avg ≥ {_cents(g['cfg']['releaseAtLeast'])}."


# ---------------- positions ----------------
def _sample_bid(rec, book):
    return None if not book else (book["up"]["bid"] if rec["dir"] == "UP" else book["down"]["bid"])


def _sample_path(rec, book, now):
    bid = _sample_bid(rec, book)
    if bid is None:
        return None
    if now - rec["lastSample"] >= PAPER["pathSampleMs"]:
        rec["path"].append(round(bid * 10) / 10)
        rec["lastSample"] = now
    return bid


def _base_record(rec):
    return {k: rec[k] for k in ("id", "sym", "tf", "strat", "dir", "conf", "strike", "entry", "tp", "start", "end", "shadow")} | {"opened": rec["opened"]}


def apply(paper: Paper, sym, tf, tf_ms, eng, sig, now, exits=None):
    """Mutates `sig` (HOLD / gate / paused texts) like the JS and returns the newly opened, visible recs."""
    exits = exits if exits is not None else []
    opened = []
    if eng.contract_strike is None or eng.contract_start is None:
        return opened
    update_gates(paper, tf, now)
    end = eng.contract_start + CONTRACT_MS
    remaining = end - now

    for s in STRAT_KEYS:
        key = f"{sym}|{tf}|{s}|{eng.contract_start}"
        rec = paper.open.get(key)
        cur = sig[s]

        if s in paper.disabled and not rec:
            sig[s] = dict(signal="WAIT", conf=None, reason="Paused — mid-market strategies are switched off.")
            continue

        if rec:
            bid = _sample_path(rec, sig.get("book"), now)
            if not rec["exited"] and rec["tp"] is not None and bid is not None and remaining > TIMING["midManageUntil"] * MIN_MS and bid >= rec["entry"] + rec["tp"]:
                rec["exited"] = dict(pnl=rec["tp"], at=now)
                rec["record"] = {**_base_record(rec), "pnl": rec["tp"], "how": "tp", "close": live_price(eng), "path": rec["path"], "settlePnl": None}
                paper.done.append(rec["record"])
                paper.rev += 1
                paper.journal.append(("close", rec["record"]))
                if not rec["shadow"]:
                    exits.append(dict(sym=sym, tf=tf, stratKey=s, kind="tp", bid=bid, level=rec["entry"] + rec["tp"]))
            cfg = paper.exit_cfg
            if cfg and not rec["shadow"] and not rec["exited"] and bid is not None and remaining > 0:
                stop_at = rec["entry"] * cfg["stopPct"]
                if not rec.get("stopAlerted") and cfg["stopPct"] > 0 and bid <= stop_at:
                    rec["stopAlerted"] = True
                    exits.append(dict(sym=sym, tf=tf, stratKey=s, kind="stop", bid=bid, level=stop_at))
                if not rec.get("tpAlerted") and rec["tp"] is None and cfg.get("tpEnd") is not None and bid >= cfg["tpEnd"]:
                    rec["tpAlerted"] = True
                    exits.append(dict(sym=sym, tf=tf, stratKey=s, kind="tp", bid=bid, level=cfg["tpEnd"]))
            if rec["exited"]:
                sig[s] = dict(signal="WAIT", conf=rec["conf"], reason=f"Target hit {_cents(rec['exited']['pnl'])} — one rec per window, done.")
            elif rec["shadow"]:
                sig[s] = dict(signal="WAIT", conf=cur["conf"], reason=_gate_reason(paper, tf, s))
            else:
                tp = f", target +{rec['tp']}¢" if rec["tp"] is not None else ""
                sig[s] = dict(signal=f"HOLD {rec['dir']}", conf=rec["conf"], reason=f"In {rec['dir']} at {rec['entry']}¢{tp}. No stop; settles at window close.")
            continue

        if not cur["signal"].startswith("BUY"):
            continue
        if cur["conf"] < PAPER["minConf"]:
            sig[s] = dict(signal="WAIT", conf=cur["conf"], reason=f"{cur['conf']}% is under the {PAPER['minConf']}% minimum.")
            continue
        held = gate_held(paper, tf, s)
        d = "UP" if cur["signal"].endswith("UP") else "DOWN"
        fresh = dict(id=key, sym=sym, tf=tf, strat=s, dir=d, conf=cur["conf"], strike=eng.contract_strike, entry=cur["ask"], tp=cur.get("tp"),
                     start=eng.contract_start, end=end, shadow=held, path=[], lastSample=0, exited=None, record=None, opened=now)
        _sample_path(fresh, sig.get("book"), now)
        paper.open[key] = fresh
        paper.journal.append(("open", _base_record(fresh)))
        if held:
            sig[s] = dict(signal="WAIT", conf=cur["conf"], reason=_gate_reason(paper, tf, s))
        else:
            opened.append(dict(stratKey=s, signal=cur["signal"], conf=cur["conf"], ask=cur.get("ask"), reason=cur.get("reason")))
    return opened


def settle(paper: Paper, engines: dict, now) -> bool:
    """engines: {(sym, tf): Engine}. Binary payout: winner 100¢, loser 0¢."""
    changed = False
    for key, rec in list(paper.open.items()):
        if now < rec["end"]:
            continue
        eng = engines.get((rec["sym"], rec["tf"]))
        price = live_price(eng) if eng else None
        if price is None:
            continue
        del paper.open[key]
        win = price > rec["strike"] if rec["dir"] == "UP" else price < rec["strike"]
        settle_pnl = 100 - rec["entry"] if win else -rec["entry"]
        if rec["exited"]:
            rec["record"]["path"] = rec["path"]
            rec["record"]["settlePnl"] = settle_pnl
            paper.journal.append(("close", rec["record"]))
        else:
            r = {**_base_record(rec), "pnl": settle_pnl, "how": "settle", "close": price, "path": rec["path"], "settlePnl": settle_pnl}
            paper.done.append(r)
            paper.journal.append(("close", r))
        paper.rev += 1
        changed = True
    return changed


# ---------------- stop-loss / take-profit optimizer ----------------
def _replay(r, sl, tp, fee):
    stop = None if sl is None else r["entry"] * sl
    target = None if tp is None else r["entry"] + tp
    path = r["path"]
    for i, bid in enumerate(path):
        if stop is not None and bid <= stop:
            fill = path[min(i + PAPER["exitLagSamples"], len(path) - 1)]
            return fill - r["entry"] - 2 * fee
        if target is not None and bid >= target:
            return tp - fee
    return r["settlePnl"] - fee


def _optimize(recs, fee):
    usable = [r for r in recs if r.get("settlePnl") is not None and r.get("path")]
    if len(usable) < PAPER["optimizerMinTrades"]:
        return None
    best = dict(sl=None, tp=None, sum=sum(r["settlePnl"] - fee for r in usable))
    for sl in PAPER["slGrid"]:
        for tp in PAPER["tpGrid"]:
            if sl is None and tp is None:
                continue
            total = sum(_replay(r, sl, tp, fee) for r in usable)
            if total > best["sum"] + 1e-9:
                best = dict(sl=sl, tp=tp, sum=total)
    return best


def summarize(paper: Paper, tf, tickers, now) -> dict:
    """Same JSON shape the browser's summarize() returns, so the existing PaperPanel renders it unchanged."""
    update_gates(paper, tf, now)
    fee = paper.fee_cents or 0
    key = f"{tf}|{paper.rev}|{','.join(tickers)}|{fee}"
    cache = paper._opt_val if paper._opt_key == key else {}
    out = {}
    for s in STRAT_KEYS:
        recs = [r for r in paper.done if r["tf"] == tf and r["strat"] == s]
        g = paper.gate[f"{tf}|{s}"]

        def row(label, gross, s=s):
            ck = f"{s}|{label}"
            if ck not in cache:
                cache[ck] = _optimize(gross, fee)
            opt = cache[ck]
            rs = _net_all(gross, fee)
            return dict(label=label, all=tally(rs), d24=tally(_within(rs, now, 24 * H)), h2=tally(_within(rs, now, 2 * H)),
                        sl=("—" if not opt else "off" if opt["sl"] is None else f"{round(opt['sl'] * 100)}%"),
                        tp=("—" if not opt else "hold" if opt["tp"] is None else f"{opt['tp']}¢"),
                        rec=None if not opt else opt["sum"])

        out[s] = dict(gate=dict(held=g["held"], hold=g["hold"], rel=g["rel"], cfg=g["cfg"]),
                      rows=[row("ALL", recs)] + [row(t, [r for r in recs if r["sym"] == t]) for t in tickers])
    paper._opt_key, paper._opt_val = key, cache
    return out

import { PAPER, CONTRACT_MS, MIN_MS, TIMING, STRAT_KEYS } from './constants';
import { livePrice } from './engine';
import { fmtCents } from './format';

const H = 3600000;

// ---------- storage (v2: records carry cents P&L; old win/loss-only v1 records are ignored) ----------
// feeCents / disabled / exitCfg are set by the hook from the settings + alert panels (see useSignalDesk); the defaults keep
// the engine behaving exactly like the fee-free, all-strategies-on original when nothing sets them (tests rely on that).
export const createPaper = (done = []) => ({
  open: new Map(), done, gate: {}, rev: 0, dirty: false, optKey: '', optVal: null,
  feeCents: 0, disabled: new Set(), exitCfg: null,
});
export function loadDone(exId) {
  try { return JSON.parse(localStorage.getItem('sd.paper2.' + exId)) || []; } catch { return []; }
}
export function saveDone(exId, paper) {
  try { localStorage.setItem('sd.paper2.' + exId, JSON.stringify(paper.done.slice(-1500))); } catch { /* noop */ }
  paper.dirty = false;
}

// ---------- stats ----------
function tally(recs) {
  const n = recs.length;
  const w = recs.filter((r) => r.pnl > 0).length;
  const sum = recs.reduce((a, r) => a + r.pnl, 0);
  const sorted = [...recs].sort((a, b) => a.end - b.end);
  let streak = 0;
  const last = n ? sorted[n - 1].pnl > 0 : null;
  for (let i = n - 1; i >= 0 && sorted[i].pnl > 0 === last; i--) streak++;
  return { n, w, pct: n ? w / n : null, sum, avg: n ? sum / n : null, streak: n ? `${streak}${last ? 'W' : 'L'}` : '—' };
}
const within = (recs, now, span) => recs.filter((r) => now - r.end <= span);

// Records are stored GROSS (before fees) so changing the fee setting re-prices the whole history. Every live record paid the
// entry fee once; a take-profit is a resting limit order (no exit fee) and settlement is free.
const net = (r, fee) => (fee ? { ...r, pnl: r.pnl - fee, settlePnl: r.settlePnl == null ? null : r.settlePnl - fee } : r);
const netAll = (recs, fee) => (fee ? recs.map((r) => net(r, fee)) : recs);

// ---------- gates (transcript) ----------
// Strategy-wide (all tickers). Judged on AVERAGE cents per contract, not win rate.
//   mid-market forks: hold when trailing 1h avg <= -2¢, release when trailing 30m avg is back up.
//   end-of-market:    hold when trailing 2h avg <= -2¢, release when trailing 1h avg > 0.
// Held strategies KEEP paper-trading in the background ("shadow" trades) — that's the only way the stats can recover and release.
export function updateGates(paper, tf, now) {
  STRAT_KEYS.forEach((s) => {
    const cfg = s === 'end' ? PAPER.gates.end : PAPER.gates.mid;
    const recs = netAll(paper.done.filter((r) => r.tf === tf && r.strat === s), paper.feeCents);
    const st = paper.gate[`${tf}|${s}`] || (paper.gate[`${tf}|${s}`] = { held: false });
    const hold = tally(within(recs, now, cfg.holdSpanMin * 60000));
    const rel = tally(within(recs, now, cfg.releaseSpanMin * 60000));
    if (!st.held) {
      if (hold.n >= PAPER.gateMinN && hold.avg <= cfg.holdBelow) st.held = true;
    } else if (rel.n >= 1 && rel.avg >= cfg.releaseAtLeast) {
      st.held = false;
    }
    st.hold = { label: cfg.holdSpanMin >= 60 ? `${cfg.holdSpanMin / 60}h` : `${cfg.holdSpanMin}m`, n: hold.n, avg: hold.avg };
    st.rel = { label: cfg.releaseSpanMin >= 60 ? `${cfg.releaseSpanMin / 60}h` : `${cfg.releaseSpanMin}m`, n: rel.n, avg: rel.avg };
    st.cfg = cfg;
  });
}
export const gateHeld = (paper, tf, s) => !!(paper.gate[`${tf}|${s}`] && paper.gate[`${tf}|${s}`].held);
function gateReason(paper, tf, s) {
  const g = paper.gate[`${tf}|${s}`];
  return `Gate closed: ${g.hold.label} avg ${fmtCents(g.hold.avg)} (n=${g.hold.n}); reopens when ${g.rel.label} avg ≥ ${fmtCents(g.cfg.releaseAtLeast)}.`;
}

// ---------- positions ----------
const sampleBid = (rec, book) => (book ? (rec.dir === 'UP' ? book.up.bid : book.down.bid) : null);

function samplePath(rec, book, now) {
  const bid = sampleBid(rec, book);
  if (bid == null) return null;
  if (now - rec.lastSample >= PAPER.pathSampleMs) {
    rec.path.push(Math.round(bid * 10) / 10);
    rec.lastSample = now;
  }
  return bid;
}

// Applies min-confidence, the gate, one-rec-per-window and the HOLD display to a fresh signal set (mutates sig).
// Returns the recs that were newly opened AND visible (these are what the signal log records). Gated recs are opened as shadow trades.
// `exits` (optional array) receives {sym, tf, stratKey, kind: 'tp'|'stop', bid, level} events for visible (non-shadow) recs.
export function apply(paper, sym, tf, tfMs, eng, sig, exits = []) {
  const opened = [];
  const now = Date.now();
  if (eng.contractStrike == null || eng.contractStart == null) return opened;
  updateGates(paper, tf, now);
  const end = eng.contractStart + CONTRACT_MS;
  const remainingMs = end - now;

  STRAT_KEYS.forEach((s) => {
    const key = `${sym}|${tf}|${s}|${eng.contractStart}`;
    const rec = paper.open.get(key);
    const cur = sig[s];

    // strategy switched off: no new recs; a rec that was already open just rides to settlement
    if (paper.disabled && paper.disabled.has(s) && !rec) {
      sig[s] = { signal: 'WAIT', conf: null, reason: 'Paused — mid-market strategies are switched off.' };
      return;
    }

    if (rec) {
      const bid = samplePath(rec, sig.book, now);
      // mid-market take-profit: watched until 1:00 remains, then it just rides to settlement
      if (!rec.exited && rec.tp != null && bid != null && remainingMs > TIMING.midManageUntil * MIN_MS && bid >= rec.entry + rec.tp) {
        rec.exited = { pnl: rec.tp, at: now };
        rec.record = { ...baseRecord(rec), pnl: rec.tp, how: 'tp', close: livePrice(eng), path: rec.path, settlePnl: null };
        paper.done.push(rec.record);
        paper.rev++; paper.dirty = true;
        if (!rec.shadow) exits.push({ sym, tf, stratKey: s, kind: 'tp', bid, level: rec.entry + rec.tp });
      }
      // exit-level cues (alerts only — they don't change the paper result): stop level, and the end-of-market take-profit price
      const cfg = paper.exitCfg;
      if (cfg && !rec.shadow && !rec.exited && bid != null && remainingMs > 0) {
        const stopAt = rec.entry * cfg.stopPct;
        if (!rec.stopAlerted && cfg.stopPct > 0 && bid <= stopAt) {
          rec.stopAlerted = true;
          exits.push({ sym, tf, stratKey: s, kind: 'stop', bid, level: stopAt });
        }
        if (!rec.tpAlerted && rec.tp == null && cfg.tpEnd != null && bid >= cfg.tpEnd) {
          rec.tpAlerted = true;
          exits.push({ sym, tf, stratKey: s, kind: 'tp', bid, level: cfg.tpEnd });
        }
      }
      if (rec.exited) {
        sig[s] = { signal: 'WAIT', conf: rec.conf, reason: `Target hit ${fmtCents(rec.exited.pnl)} — one rec per window, done.` };
      } else if (rec.shadow) {
        sig[s] = { signal: 'WAIT', conf: cur.conf, reason: gateReason(paper, tf, s) };
      } else {
        sig[s] = { signal: `HOLD ${rec.dir}`, conf: rec.conf, reason: `In ${rec.dir} at ${rec.entry}¢${rec.tp != null ? `, target +${rec.tp}¢` : ''}. No stop; settles at window close.` };
      }
      return;
    }

    if (!cur.signal.startsWith('BUY')) return; // WAIT / HOLD-above-98 are display only
    if (cur.conf < PAPER.minConf) {
      sig[s] = { signal: 'WAIT', conf: cur.conf, reason: `${cur.conf}% is under the ${PAPER.minConf}% minimum.` };
      return;
    }
    const held = gateHeld(paper, tf, s);
    const dir = cur.signal.endsWith('UP') ? 'UP' : 'DOWN';
    const fresh = {
      id: `${key}`, sym, tf, strat: s, dir, conf: cur.conf, strike: eng.contractStrike, entry: cur.ask, tp: cur.tp,
      start: eng.contractStart, end, shadow: held, path: [], lastSample: 0, exited: null, record: null, opened: now,
    };
    samplePath(fresh, sig.book, now);
    paper.open.set(key, fresh);
    if (held) {
      sig[s] = { signal: 'WAIT', conf: cur.conf, reason: gateReason(paper, tf, s) };
    } else {
      opened.push({ stratKey: s, signal: cur.signal, conf: cur.conf });
    }
  });
  return opened;
}

const baseRecord = (rec) => ({
  id: rec.id, sym: rec.sym, tf: rec.tf, strat: rec.strat, dir: rec.dir, conf: rec.conf, strike: rec.strike,
  entry: rec.entry, tp: rec.tp, start: rec.start, end: rec.end, shadow: rec.shadow,
});

// Settles every rec whose window has closed. Binary payout: a winning contract pays 100¢, a losing one 0¢.
// A mid-market rec that already took profit keeps running as a "ghost" only to record its full price path for the optimizer.
export function settle(paper, engines, now) {
  let changed = false;
  paper.open.forEach((rec, key) => {
    if (now < rec.end) return;
    const eng = engines[rec.sym] && engines[rec.sym][rec.tf];
    const price = eng ? livePrice(eng) : null;
    if (price == null) return;
    paper.open.delete(key);
    const up = price >= rec.strike; // tie settles UP (same rule as backend/app/settlement.py)
    const win = rec.dir === 'UP' ? up : !up;
    const settlePnl = win ? 100 - rec.entry : -rec.entry;
    if (rec.exited) {
      rec.record.path = rec.path;
      rec.record.settlePnl = settlePnl;
    } else {
      paper.done.push({ ...baseRecord(rec), pnl: settlePnl, how: 'settle', close: price, path: rec.path, settlePnl });
    }
    paper.rev++; paper.dirty = true;
    changed = true;
  });
  return changed;
}

// ---------- stop-loss / take-profit optimizer (transcript) ----------
// Replays every completed result's recorded price path under each candidate rule and keeps the rule with the best total P&L.
// SL is a % of the entry price (e.g. 80% of a 90¢ entry = stop at 72¢), TP is cents above entry; null = "off" / "hold".
// A stop is "sell at the best price above 1¢": it fills at the bid you can actually hit when you press (the sampled bid at/after
// the trigger, see PAPER.exitLagSamples), which is at or below the stop price — not at the stop price. A stop pays the entry fee
// AND an exit fee (market-style sell); a take-profit is a resting limit (entry fee only); holding pays the entry fee only.
function replay(r, sl, tp, fee) {
  const stop = sl == null ? null : r.entry * sl;
  const target = tp == null ? null : r.entry + tp;
  for (let i = 0; i < r.path.length; i++) {
    const bid = r.path[i];
    if (stop != null && bid <= stop) {
      const fill = r.path[Math.min(i + PAPER.exitLagSamples, r.path.length - 1)];
      return fill - r.entry - 2 * fee;
    }
    if (target != null && bid >= target) return tp - fee;
  }
  return r.settlePnl - fee;
}
function optimize(recs, fee) {
  const usable = recs.filter((r) => r.settlePnl != null && r.path && r.path.length);
  if (usable.length < PAPER.optimizerMinTrades) return null;
  let best = { sl: null, tp: null, sum: replay0(usable, fee) };
  PAPER.slGrid.forEach((sl) => PAPER.tpGrid.forEach((tp) => {
    if (sl == null && tp == null) return;
    const sum = usable.reduce((a, r) => a + replay(r, sl, tp, fee), 0);
    if (sum > best.sum + 1e-9) best = { sl, tp, sum };
  }));
  return best;
}
const replay0 = (recs, fee) => recs.reduce((a, r) => a + r.settlePnl - fee, 0);

export function summarize(paper, tf, tickers, now) {
  updateGates(paper, tf, now);
  const fee = paper.feeCents || 0;
  const key = `${tf}|${paper.rev}|${tickers.join(',')}|${fee}`;
  const cache = paper.optKey === key ? paper.optVal : {};
  const out = {};
  STRAT_KEYS.forEach((s) => {
    const recs = paper.done.filter((r) => r.tf === tf && r.strat === s);
    const g = paper.gate[`${tf}|${s}`];
    const row = (label, gross) => {
      const ck = `${s}|${label}`;
      if (!(ck in cache)) cache[ck] = optimize(gross, fee);
      const opt = cache[ck];
      const rs = netAll(gross, fee);
      return {
        label, all: tally(rs), d24: tally(within(rs, now, 24 * H)), h2: tally(within(rs, now, 2 * H)),
        sl: opt ? (opt.sl == null ? 'off' : `${Math.round(opt.sl * 100)}%`) : '—',
        tp: opt ? (opt.tp == null ? 'hold' : `${opt.tp}¢`) : '—',
        rec: opt ? opt.sum : null,
      };
    };
    out[s] = {
      gate: { held: g.held, hold: g.hold, rel: g.rel, cfg: g.cfg },
      rows: [row('ALL', recs), ...tickers.map((t) => row(t, recs.filter((r) => r.sym === t)))],
    };
  });
  paper.optKey = key;
  paper.optVal = cache;
  return out;
}

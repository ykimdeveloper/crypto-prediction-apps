// Turns one per-symbol snapshot from the FastAPI /ws/desk stream into the same "card" object useSignalDesk builds locally,
// so SymbolCard / PaperPanel / SignalLog render both modes identically. Pure functions (no React) so they can be unit-tested.
import { CONTRACT_MS } from './constants';
import { fmt, fmtRemaining } from './format';

const WAIT = { signal: 'WAIT', conf: null, reason: 'Loading…' };

export const emptyCard = (symbol, quote) => ({
  symbol, quote, spot: '—', strike: '—', dist: '—', book: null, spotBook: null, stoch: '—', bb: '—', atr: '—',
  lean: '—', story: '—', atrState: '—', module: '—', closesIn: '—', rev: WAIT, mom: WAIT, end: WAIT, failed: false, ml: null, venues: null,
});

const gate = { held: false, hold: { label: '—', n: 0, avg: null }, rel: { label: '—', n: 0, avg: null }, cfg: {} };
export const EMPTY_PAPER = { rev: { gate, rows: [] }, mom: { gate, rows: [] }, end: { gate, rows: [] } };

// consolidated feeds: one chip per selected exchange (price, offset from the consolidated price in bps, live/stale/error)
const shapeVenues = (vs) => vs.map((v) => ({
  exchange: v.exchange, status: v.status, symbol: v.symbol, error: v.error || null,
  price: v.price != null ? fmt(v.price) : null,
  devBps: v.devBps != null ? `${v.devBps >= 0 ? '+' : ''}${v.devBps.toFixed(1)}bp` : null,
  rejected: v.rejected || 0,
}));

export function serverCard(snap, quote, exName, pair, now = Date.now()) {
  if (!snap || !snap.ind || !snap.sig) {
    const base = emptyCard(snap ? snap.symbol : '', quote);
    if (snap && snap.failed) base.failed = true;
    if (snap && snap.failed) base.dist = snap.venues || snap.error ? `no data — ${snap.error || 'waiting for the exchanges'}` : `no data — ${pair} may not be listed on ${exName}`;
    if (snap && snap.venues) base.venues = shapeVenues(snap.venues);
    return base;
  }
  const { ind, sig } = snap;
  const price = snap.price;
  const strike = snap.strike;
  const failed = !!snap.failed;
  const arrow = ind.stoch && (ind.stoch.traj === 'Rising' ? '↑' : ind.stoch.traj === 'Falling' ? '↓' : '→');
  return {
    symbol: snap.symbol,
    quote,
    failed,
    spot: fmt(price),
    strike: fmt(strike),
    dist: failed && price == null
      ? `no data — ${pair} may not be listed on ${exName}`
      : price != null && strike != null
      ? `${fmt(Math.abs(price - strike))} ${price - strike >= 0 ? 'above' : 'below'} strike` : '—',
    book: sig.book || null,
    spotBook: snap.spotBook ? `${fmt(snap.spotBook.bid)} / ${fmt(snap.spotBook.ask)}` : null,
    stoch: ind.stoch ? `${ind.stoch.k.toFixed(0)} ${arrow}` : '—',
    bb: ind.bb ? ind.bb.state.replace('_', ' ') : '—',
    atr: ind.atr
      ? `${fmt(ind.atr.atr)}${ind.atr.state === 'Quiet/Asleep' ? ' (asleep)' : ''}${sig.story ? ' · ' + (sig.story === 'Mean Reversion' ? 'revert' : sig.story === 'Momentum Continuation' ? 'mom' : 'mixed') + ' ' + (sig.lean > 0.05 ? '↑' : sig.lean < -0.05 ? '↓' : '→') : ''}`
      : '—',
    lean: sig.story ? (sig.lean >= 0 ? '+' : '') + sig.lean.toFixed(2) : '—',
    story: sig.story || '—',
    atrState: ind.atr ? (ind.atr.state === 'Quiet/Asleep' ? 'Asleep' : 'Active') : '—',
    module: sig.module,
    closesIn: snap.contractStart != null ? fmtRemaining(snap.contractStart + CONTRACT_MS - now) : '—',
    rev: sig.rev, mom: sig.mom, end: sig.end,
    ml: snap.mlPUp != null ? `${Math.round(snap.mlPUp * 100)}%` : null,
    policy: snap.policy && snap.policy.mode === 'gated' ? snap.policy : null,
    venues: snap.venues ? shapeVenues(snap.venues) : null,
  };
}

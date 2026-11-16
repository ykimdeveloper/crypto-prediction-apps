import { test, expect, describe, afterAll } from 'vitest';
import { createEngine } from '../src/lib/engine.js';
import { computeModuleSignals } from '../src/lib/strategy.js';
import { createPaper, apply, summarize, updateGates, gateHeld } from '../src/lib/paper.js';
import { PAPER, CONTRACT_MS } from '../src/lib/constants.js';
import { DEFAULT_ALERTS, shouldAlertExit } from '../src/lib/alerts.js';

const realNow = Date.now;
afterAll(() => { Date.now = realNow; });
const NOW = Date.UTC(2026, 8, 27, 12, 7, 0); // 12:07 -> 8:00 left in the 12:00-12:15 window
Date.now = () => NOW;

// ---- records store GROSS pnl; fee is applied when reading ----
const mk = (i, over = {}) => ({ sym: 'BTC', tf: '1m', strat: 'end', dir: 'UP', entry: 90, pnl: 8, end: NOW - i * 60000, path: [], settlePnl: 8, how: 'settle', ...over });

describe('fee-inclusive paper P&L', () => {
  test('default (fee 0) leaves stats untouched', () => {
    const p = createPaper();
    for (let i = 0; i < 4; i++) p.done.push(mk(i));
    const s = summarize(p, '1m', ['BTC'], NOW);
    expect(s.end.rows[0].all.avg).toBe(8);
  });
  test('fee is subtracted from every record and re-prices history when changed', () => {
    const p = createPaper();
    for (let i = 0; i < 4; i++) p.done.push(mk(i));
    p.feeCents = 1;
    expect(summarize(p, '1m', ['BTC'], NOW).end.rows[0].all.avg).toBe(7);
    p.feeCents = 3;
    expect(summarize(p, '1m', ['BTC'], NOW).end.rows[0].all.avg).toBe(5);
  });
  test('a thin win can become a net loss once the fee is charged', () => {
    const p = createPaper();
    p.done.push(mk(1, { pnl: 0.5, settlePnl: 0.5 }));
    p.feeCents = 0.8;
    const r = summarize(p, '1m', ['BTC'], NOW).end.rows[0].all;
    expect(r.w).toBe(0);
    expect(r.sum).toBeLessThan(0);
  });
  test('the gate reads net-of-fee results', () => {
    const p = createPaper();
    for (let i = 0; i < 6; i++) p.done.push(mk(i * 5 + 1, { pnl: -1, settlePnl: -1 })); // gross -1¢ avg
    updateGates(p, '1m', NOW);
    expect(gateHeld(p, '1m', 'end')).toBe(false); // -1 > -2
    p.feeCents = 1.5; // net -2.5
    updateGates(p, '1m', NOW);
    expect(gateHeld(p, '1m', 'end')).toBe(true);
  });
});

describe('stop-loss optimizer fills at the bid, not the stop price', () => {
  const ledger = (fee) => {
    const p = createPaper();
    p.feeCents = fee;
    // entry 90, bid path 80 -> 70 -> 40, settled as a 90¢ loss. A 90% stop (81¢) triggers on the first sample, at an 80¢ bid.
    for (let i = 0; i < 8; i++) p.done.push(mk(i, { pnl: -90, settlePnl: -90, path: [80, 70, 40] }));
    return summarize(p, '1m', ['BTC'], NOW).end.rows[0];
  };
  test('stop exit = sampled bid − entry (−10¢), not stop price − entry (−9¢)', () => {
    const r = ledger(0);
    expect(r.sl).toBe('90%');
    expect(r.tp).toBe('hold');
    expect(r.rec).toBe(8 * -10);
  });
  test('a stop exit pays the entry fee and an exit fee', () => {
    expect(ledger(1).rec).toBe(8 * -12);
  });
  test('the stop grid can express the 75% the reference bot landed on', () => {
    expect(PAPER.slGrid).toContain(0.75);
  });
});

describe('exit alerts', () => {
  const start = Math.floor(NOW / CONTRACT_MS) * CONTRACT_MS;
  const eng = { contractStart: start, contractStrike: 100, current: { c: 101 }, candles: [] };
  const wait = { signal: 'WAIT', conf: null, reason: '' };
  const book = (bid) => ({ up: { bid, ask: bid + 1 }, down: { bid: 100 - bid - 1, ask: 100 - bid } });
  const sigFor = (bid, extra = {}) => ({ rev: wait, mom: wait, end: { signal: 'BUY UP', conf: 90, ask: 90, tp: null, reason: '' }, book: book(bid), ...extra });

  test('stop level fires once when the bid falls to 75% of entry', () => {
    const p = createPaper(); p.exitCfg = { stopPct: 0.75, tpEnd: 99 };
    const ex = [];
    expect(apply(p, 'BTC', '1m', 60000, eng, sigFor(89.5), ex).length).toBe(1); // opens at 90¢
    expect(ex.length).toBe(0);
    apply(p, 'BTC', '1m', 60000, eng, sigFor(70), ex);
    expect(ex.length).toBe(0); // 70 > 67.5
    apply(p, 'BTC', '1m', 60000, eng, sigFor(66), ex);
    expect(ex.length).toBe(1);
    expect(ex[0].kind).toBe('stop');
    apply(p, 'BTC', '1m', 60000, eng, sigFor(60), ex);
    expect(ex.length).toBe(1); // once per position
  });
  test('end-of-market take-profit level fires when the bid reaches the price you set', () => {
    const p = createPaper(); p.exitCfg = { stopPct: 0.75, tpEnd: 99 };
    const ex = [];
    apply(p, 'BTC', '1m', 60000, eng, sigFor(89.5), ex);
    apply(p, 'BTC', '1m', 60000, eng, sigFor(98), ex);
    expect(ex.length).toBe(0);
    apply(p, 'BTC', '1m', 60000, eng, sigFor(99.2), ex);
    expect(ex.length).toBe(1);
    expect(ex[0].kind).toBe('tp');
  });
  test('alerts never change the paper result (end-of-market still holds to settlement)', () => {
    const p = createPaper(); p.exitCfg = { stopPct: 0.75, tpEnd: 99 };
    apply(p, 'BTC', '1m', 60000, eng, sigFor(89.5), []);
    apply(p, 'BTC', '1m', 60000, eng, sigFor(60), []);
    expect(p.done.length).toBe(0);
  });
  test('no exit config => no exit events', () => {
    const p = createPaper(); const ex = [];
    apply(p, 'BTC', '1m', 60000, eng, sigFor(89.5), ex);
    apply(p, 'BTC', '1m', 60000, eng, sigFor(10), ex);
    expect(ex.length).toBe(0);
  });
  test('shouldAlertExit honors the master switch, the exits switch, strategy and ticker filters', () => {
    const on = { ...DEFAULT_ALERTS, enabled: true };
    const rec = { sym: 'BTC', stratKey: 'end', kind: 'stop' };
    expect(shouldAlertExit(on, rec)).toBe(true);
    expect(shouldAlertExit(DEFAULT_ALERTS, rec)).toBe(false);
    expect(shouldAlertExit({ ...on, exits: { ...on.exits, enabled: false } }, rec)).toBe(false);
    expect(shouldAlertExit({ ...on, muted: ['BTC'] }, rec)).toBe(false);
  });
});

describe('mid-market toggle', () => {
  const start = Math.floor(NOW / CONTRACT_MS) * CONTRACT_MS;
  const eng = { contractStart: start, contractStrike: 100, current: { c: 101 }, candles: [] };
  const buy = (k) => ({ signal: 'BUY UP', conf: 90, ask: 60, tp: 20, reason: '' });
  const sig = () => ({ rev: buy(), mom: buy(), end: { signal: 'BUY UP', conf: 90, ask: 90, tp: null, reason: '' }, book: { up: { bid: 59, ask: 60 }, down: { bid: 40, ask: 41 } } });
  test('on: all three open', () => {
    const p = createPaper();
    expect(apply(p, 'BTC', '1m', 60000, eng, sig()).length).toBe(3);
  });
  test('off: mid forks are paused, end-of-market is unaffected', () => {
    const p = createPaper(); p.disabled = new Set(['rev', 'mom']);
    const s = sig();
    const opened = apply(p, 'BTC', '1m', 60000, eng, s);
    expect(opened.length).toBe(1);
    expect(opened[0].stratKey).toBe('end');
    expect(s.rev.signal).toBe('WAIT');
    expect(s.mom.signal).toBe('WAIT');
  });
});

describe('end-of-market shows lean + confidence before the 8:00 window opens', () => {
  const build = (minLeft) => {
    const t = start0 + (15 - minLeft) * 60000;
    Date.now = () => t;
    const eng = createEngine(60000);
    eng.candles = Array.from({ length: 60 }, (_, i) => ({ t: t - (70 - i) * 60000, o: 100, h: 101, l: 99, c: 100, v: 1 }));
    eng.ind = { bb: { mid: 100, upper: 102, lower: 98, state: 'Mid_Band' }, stoch: { k: 55, traj: 'Rising', state: 'Mixed' }, atr: { atr: 0.2, state: 'Active' } };
    eng.contractStart = start0; eng.contractStrike = 100; eng.lastTickAt = t;
    eng.current = { t: Math.floor(t / 60000) * 60000, o: 103, h: 103, l: 103, c: 103, v: 1 };
    return computeModuleSignals(eng, 60000);
  };
  const start0 = Math.floor(NOW / CONTRACT_MS) * CONTRACT_MS;
  test('11:00 left: WAIT, but with a side and a confidence', () => {
    const sig = build(11);
    expect(sig.end.signal).toBe('WAIT');
    expect(sig.end.conf).toBeGreaterThan(0);
    expect(sig.end.reason).toMatch(/Leaning UP/);
  });
  test('7:00 left: window is open (BUY or HOLD), still has confidence', () => {
    const sig = build(7);
    expect(sig.end.conf).toBeGreaterThan(0);
    expect(sig.end.reason).not.toMatch(/buys open at/);
  });
  test('1:00 left: closed', () => {
    expect(build(1).end.reason).toMatch(/Closed/);
  });
});

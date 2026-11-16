import { test, expect, describe, beforeAll, afterAll } from 'vitest';
import { createEngine, updateEngineFromKline, setEngineCandles, evaluateAndLog } from '../src/lib/engine.js';
import { createPaper, apply, settle, summarize, updateGates, gateHeld } from '../src/lib/paper.js';
import { TIMING, PRICE, PAPER } from '../src/lib/constants.js';

// deterministic RNG
let seed = 12345;
const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

const realNow = Date.now;
let NOW;
let entries, paper, stats;

// ---- run the 14h simulation once, share the result across every test below ----
describe('14h simulated session (4 symbols, regime-switching volatility)', () => {
  beforeAll(() => {
    NOW = Date.UTC(2026, 8, 27, 12, 0, 0);
    Date.now = () => NOW;

    const SYMS = ['BTC', 'ETH', 'SOL', 'XRP'];
    const start = { BTC: 78000, ETH: 2470, SOL: 103, XRP: 1.38 };
    const engines = {}; const price = {}; const minuteBar = {};
    SYMS.forEach((s) => {
      const eng = createEngine(60000); engines[s] = { '1m': eng }; price[s] = start[s];
      const cs = []; let p = price[s];
      for (let i = 120; i > 0; i--) { const t = NOW - i * 60000; const o = p; p = p * (1 + 0.0004 * gauss()); cs.push({ t, o, h: Math.max(o, p) * 1.0002, l: Math.min(o, p) * 0.9998, c: p, v: 1 }); }
      price[s] = p; setEngineCandles(eng, cs);
    });

    paper = createPaper();
    entries = [];
    const seenOpen = new Set();

    const HOURS = 14;
    for (let sec = 0; sec < HOURS * 3600; sec++) {
      NOW += 1000;
      SYMS.forEach((s) => {
        const regime = Math.floor(sec / 1800) % 3;
        const vol = [0.00012, 0.00026, 0.00018][regime];
        const drift = regime === 1 ? 0.00001 * (Math.floor(sec / 1800) % 2 ? 1 : -1) : 0;
        price[s] *= (1 + drift + vol * gauss());
        const bucket = Math.floor(NOW / 60000) * 60000;
        let b = minuteBar[s];
        if (!b || b.t !== bucket) b = minuteBar[s] = { t: bucket, o: price[s], h: price[s], l: price[s], c: price[s], v: 1 };
        b.h = Math.max(b.h, price[s]); b.l = Math.min(b.l, price[s]); b.c = price[s];
        const closing = (NOW + 1000) >= bucket + 60000;
        updateEngineFromKline(engines[s]['1m'], { ...b, x: closing });
        if (closing) delete minuteBar[s];
        const eng = engines[s]['1m'];
        const { sig } = evaluateAndLog(eng, 60000);
        const before = new Set(paper.open.keys());
        apply(paper, s, '1m', 60000, eng, sig);
        paper.open.forEach((rec, key) => { if (!before.has(key) && !seenOpen.has(key)) { seenOpen.add(key); entries.push({ ...rec, remMinAtOpen: (rec.end - NOW) / 60000 }); } });
      });
      settle(paper, engines, NOW);
    }
    stats = summarize(paper, '1m', SYMS, NOW);
  });

  afterAll(() => { Date.now = realNow; });

  const by = (s) => entries.filter((e) => e.strat === s);

  test('some trades happened for each strategy', () => {
    for (const s of ['rev', 'mom', 'end']) expect(by(s).length).toBeGreaterThan(0);
  });

  test('mid-market recs only open with >= 5:00 left', () => {
    const mid = entries.filter((e) => e.strat !== 'end');
    for (const e of mid) expect(e.remMinAtOpen).toBeGreaterThanOrEqual(TIMING.midEntryMinRemaining - 0.02);
  });

  test('mid-market entry ask >= 35¢ minimum', () => {
    const mid = entries.filter((e) => e.strat !== 'end');
    for (const e of mid) expect(e.entry).toBeGreaterThanOrEqual(PRICE.minEntry);
  });

  test('MMR entries inside 35–50¢ band', () => {
    for (const e of by('rev')) { expect(e.entry).toBeGreaterThanOrEqual(PRICE.mmrBand[0]); expect(e.entry).toBeLessThanOrEqual(PRICE.mmrBand[1]); }
  });

  test('MMC entries inside 50–78¢ band', () => {
    for (const e of by('mom')) { expect(e.entry).toBeGreaterThanOrEqual(PRICE.mmcBand[0]); expect(e.entry).toBeLessThanOrEqual(PRICE.mmcBand[1]); }
  });

  test('EOM entries between 8:00 and 2:00 left', () => {
    for (const e of by('end')) { expect(e.remMinAtOpen).toBeLessThanOrEqual(TIMING.eomMaxRemaining + 0.02); expect(e.remMinAtOpen).toBeGreaterThanOrEqual(TIMING.eomMinRemaining - 0.02); }
  });

  test('EOM entry ask inside 85–98¢', () => {
    for (const e of by('end')) { expect(e.entry).toBeGreaterThanOrEqual(PRICE.eomBand[0]); expect(e.entry).toBeLessThanOrEqual(PRICE.eomBand[1]); }
  });

  test('every entry had >= 80% confidence', () => {
    for (const e of entries) expect(e.conf).toBeGreaterThanOrEqual(PAPER.minConf);
  });

  test('one rec per symbol/strategy/window', () => {
    const keys = entries.map((e) => `${e.sym}|${e.strat}|${e.start}`);
    expect(new Set(keys).size).toBe(keys.length);
  });

  test('take-profit exits book exactly the target (+15 MMR / +20 MMC)', () => {
    const tpRecs = paper.done.filter((r) => r.how === 'tp');
    expect(tpRecs.length).toBeGreaterThan(0);
    for (const r of tpRecs) expect(r.pnl).toBe(r.strat === 'rev' ? PRICE.tpMmr : PRICE.tpMmc);
  });

  test('settlement is binary: win = 100−entry, loss = −entry', () => {
    const settled = paper.done.filter((r) => r.how === 'settle');
    expect(settled.length).toBeGreaterThan(0);
    for (const r of settled) expect(r.pnl === 100 - r.entry || r.pnl === -r.entry).toBe(true);
  });

  test('EOM never exits early (no stop, holds to settlement)', () => {
    for (const r of paper.done.filter((r) => r.strat === 'end')) expect(r.how).toBe('settle');
  });

  test('TP ghosts got their full path + settle P&L patched in', () => {
    const finishedTp = paper.done.filter((r) => r.how === 'tp' && r.end < NOW - 5000);
    expect(finishedTp.length).toBeGreaterThan(0);
    for (const r of finishedTp) { expect(r.settlePnl).not.toBeNull(); expect(r.path.length).toBeGreaterThan(3); }
  });

  test('optimizer ran for strategies with enough history', () => {
    const withOpt = ['rev', 'mom', 'end'].map((s) => stats[s].rows[0]).filter((r) => r.rec != null);
    expect(withOpt.length).toBeGreaterThan(0);
  });

  for (const s of ['rev', 'mom', 'end']) {
    test(`${s}: REC PNL is never worse than plain hold-to-settle`, () => {
      const r = stats[s].rows[0];
      if (r.rec == null) return; // not enough history yet, nothing to assert
      const base = paper.done.filter((x) => x.strat === s && x.settlePnl != null && x.path && x.path.length).reduce((a, x) => a + x.settlePnl, 0);
      expect(r.rec).toBeGreaterThanOrEqual(base - 1e-9);
    });
  }
});

// ---- gate: trip, shadow-trade, release (synthetic ledgers, independent of the simulation above) ----
describe('mid-market gate (1h <= -2¢ holds, 30m >= +2.5¢ releases)', () => {
  afterAll(() => { Date.now = realNow; });
  const T = Date.UTC(2026, 8, 27, 12, 0, 0);
  const mk = (strat, pnl, minsAgo, sym = 'BTC') => ({ sym, tf: '1m', strat, pnl, end: T - minsAgo * 60000, entry: 90, path: [], settlePnl: pnl });
  const p2 = createPaper();

  test('trips: 1h avg <= -2¢ with n>=5', () => {
    [-20, -10, 5, -15, 4, -10].forEach((v, i) => p2.done.push(mk('rev', v, 50 - i * 5)));
    updateGates(p2, '1m', T);
    expect(gateHeld(p2, '1m', 'rev')).toBe(true);
    expect(p2.gate['1m|rev'].hold.avg).toBeLessThanOrEqual(-2);
  });

  test('other strategies unaffected (gate is per strategy)', () => {
    expect(gateHeld(p2, '1m', 'mom')).toBe(false);
    expect(gateHeld(p2, '1m', 'end')).toBe(false);
  });

  test('stays held while the 30m average is still under the release line', () => {
    p2.done.push(mk('rev', -3, 4));
    updateGates(p2, '1m', T);
    expect(gateHeld(p2, '1m', 'rev')).toBe(true);
  });

  test('releases when the trailing 30m average recovers (shadow trades made that possible)', () => {
    [15, 20, 15].forEach((v, i) => p2.done.push(mk('rev', v, 3 - i)));
    updateGates(p2, '1m', T);
    expect(gateHeld(p2, '1m', 'rev')).toBe(false);
    expect(p2.gate['1m|rev'].rel.avg).toBeGreaterThanOrEqual(2.5);
  });
});

describe('end-of-market gate (2h <= -2¢ holds, 1h > 0¢ releases)', () => {
  afterAll(() => { Date.now = realNow; });
  const T = Date.UTC(2026, 8, 27, 12, 0, 0);
  const mk = (strat, pnl, minsAgo, sym = 'BTC') => ({ sym, tf: '1m', strat, pnl, end: T - minsAgo * 60000, entry: 90, path: [], settlePnl: pnl });
  const p3 = createPaper();

  test('trips on a 2h avg <= -2¢', () => {
    [-60, -20, -30, 10, -40, 5, -25].forEach((v, i) => p3.done.push(mk('end', v, 100 - i * 10)));
    updateGates(p3, '1m', T);
    expect(gateHeld(p3, '1m', 'end')).toBe(true);
  });

  test('stays held while the trailing 1h avg is still negative', () => {
    [8, 9, 8].forEach((v, i) => p3.done.push(mk('end', v, 20 - i * 5)));
    updateGates(p3, '1m', T);
    expect(gateHeld(p3, '1m', 'end')).toBe(true);
    expect(p3.gate['1m|end'].rel.avg).toBeLessThan(0);
  });

  test('releases once the trailing 1h avg turns positive', () => {
    [9, 8, 9, 10, 8, 9, 8, 9, 9, 10].forEach((v, i) => p3.done.push(mk('end', v, 18 - i)));
    updateGates(p3, '1m', T);
    expect(gateHeld(p3, '1m', 'end')).toBe(false);
  });
});

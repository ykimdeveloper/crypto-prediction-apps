import { test, expect, describe, afterAll } from 'vitest';
import { computeModuleSignals, normalCdf } from '../src/lib/strategy.js';
import { createEngine } from '../src/lib/engine.js';
import { MID, EOM_CONF_SIGMA } from '../src/lib/constants.js';

// Rules from the v0.2 video: momentum-only mid-market, 6:00 minimum, price already slightly on the lean's side of the
// strike (not exhausted), a high chance of the +20¢, and an end-of-market indicator nudge that fades with time.
const realNow = Date.now;
afterAll(() => { Date.now = realNow; });
const NOW = 1_000_000 * 900_000 + 7 * 60_000;

// z = signed distance from the strike in sigmas of the remaining path (ATR * sqrt(candles left))
function sig({ rem, z, k, traj, sstate = 'Mixed', atr = 0.5, astate = 'Active', bbw = 2 }) {
  Date.now = () => NOW;
  const eng = createEngine(60000);
  eng.contractStart = NOW - (15 - rem) * 60000;
  eng.contractStrike = 100;
  eng.current = { t: eng.contractStart, o: 100, h: 100, l: 100, c: 100 + z * atr * Math.sqrt(rem), v: 1 };
  eng.lastTickAt = NOW;
  eng.ind = {
    bb: { mid: 100, upper: 100 + bbw, lower: 100 - bbw, state: 'Mid_Band' },
    stoch: { k, traj, state: sstate },
    atr: { atr, state: astate },
  };
  return computeModuleSignals(eng, 60000);
}
const MOM_UP = { rem: 10, z: 0.4, k: 70, traj: 'Rising' };

describe('mid-market is momentum-only', () => {
  const REV_SETUP = { rem: 10, z: -0.506, k: 10, traj: 'Rising', sstate: 'Bounce_Bottom', bbw: 1 };

  test('a textbook snapback is still classified, but MID REV will not buy', () => {
    const s = sig(REV_SETUP);
    expect(s.story).toBe('Mean Reversion');
    expect(s.lean).toBeGreaterThan(0);
    expect(s.rev.signal).toBe('WAIT');
    expect(s.rev.reason).toMatch(/off in v0\.2/);
    expect(s.mom.signal).toBe('WAIT');
  });

  test('MID.revEnabled = true brings the reversion fork back', () => {
    MID.revEnabled = true;
    try { expect(sig(REV_SETUP).rev.signal).toBe('BUY UP'); } finally { MID.revEnabled = false; }
  });

  test('a momentum setup buys', () => {
    const s = sig(MOM_UP);
    expect(s.story).toBe('Momentum Continuation');
    expect(s.mom.signal).toBe('BUY UP');
    expect(s.mom.pHit).toBeGreaterThanOrEqual(MID.minHitProb);
  });
});

describe('needs about six minutes left', () => {
  test('6:00 is allowed, just under is not', () => {
    expect(sig({ ...MOM_UP, rem: 6.0 }).mom.signal).toBe('BUY UP');
    const s = sig({ ...MOM_UP, rem: 5.9 });
    expect(s.mom.signal).toBe('WAIT');
    expect(s.mom.reason).toMatch(/Under 6:00 left/);
  });
});

describe('price already a little on the lean side of the strike, not exhausted', () => {
  test('barely on the side (or still on the wrong side) -> wait', () => {
    for (const z of [-0.3, 0.0, 0.05]) {
      const s = sig({ ...MOM_UP, z });
      expect(s.mom.signal).toBe('WAIT');
      expect(s.mom.reason).toMatch(/isn't on the UP side/);
    }
  });
  test('already run a long way -> wait', () => {
    const s = sig({ ...MOM_UP, z: 1.3 });
    expect(s.mom.signal).toBe('WAIT');
    expect(s.mom.reason).toMatch(/already ran/);
  });
  test('the down side mirrors the up side', () => {
    const s = sig({ rem: 10, z: -0.4, k: 30, traj: 'Falling' });
    expect(s.mom.signal).toBe('BUY DOWN');
  });
});

describe('the +20¢ has to be likely enough', () => {
  test('raising the bar above what this entry offers -> wait', () => {
    const before = sig(MOM_UP).mom.pHit;
    MID.minHitProb = before + 0.02;
    try {
      const s = sig(MOM_UP);
      expect(s.mom.signal).toBe('WAIT');
      expect(s.mom.reason).toMatch(/% likely/);
    } finally { MID.minHitProb = 0.72; }
  });
  test('an unreachable bar blocks every mid-market entry', () => {
    MID.minHitProb = 0.95;
    try { for (const z of [0.2, 0.4, 0.7, 0.9]) expect(sig({ ...MOM_UP, z }).mom.signal).toBe('WAIT'); } finally { MID.minHitProb = 0.72; }
  });
});

describe('asleep ATR still blocks', () => {
  test('quiet market -> wait', () => {
    const s = sig({ ...MOM_UP, astate: 'Quiet/Asleep' });
    expect(s.mom.signal).toBe('WAIT');
    expect(s.mom.reason).toMatch(/asleep/);
  });
});

describe('end-of-market: the indicator nudge fades as time runs out', () => {
  // price above the strike (so END favors UP) while the indicators lean DOWN: the nudge can only pull confidence down
  const disagree = (rem) => sig({ rem, z: 0.5, k: 92, traj: 'Falling', sstate: 'Rollover_Top' }).end.conf;
  const stat = normalCdf(0.5 / EOM_CONF_SIGMA) * 100;

  test('confidence recovers toward pure cushion-vs-noise as the window closes', () => {
    expect(disagree(2)).toBeGreaterThan(disagree(5));
    expect(disagree(5)).toBeGreaterThan(disagree(8));
  });
  test('no nudge left at 2:00: confidence is exactly the statistical part', () => {
    expect(disagree(2)).toBe(Math.round(stat));
  });
  test('full nudge at 8:00 (the old fixed blend)', () => {
    expect(disagree(8)).toBeLessThan(Math.round(stat));
  });
});

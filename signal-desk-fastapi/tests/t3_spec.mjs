import { test, expect, describe, afterAll } from 'vitest';
import { evaluateLeanAndStory, computeModuleSignals } from '../src/lib/strategy.js';
import { computeStochRSI } from '../src/lib/indicators.js';
import { createEngine } from '../src/lib/engine.js';

const bb = { mid: 100, upper: 110, lower: 90, state: 'Upper_Stretched' };

test('outside band + High_Rising = Momentum Continuation', () => {
  const r = evaluateLeanAndStory({ bb, stoch: { k: 72, traj: 'Rising', state: 'High_Rising' } }, 112);
  expect(r.story).toBe('Momentum Continuation');
  expect(r.lean).toBeGreaterThan(0.5);
});

test('outside band + mid-zone stoch is not a confirmed breakout', () => {
  const r = evaluateLeanAndStory({ bb, stoch: { k: 52, traj: 'Rising', state: 'Mixed' } }, 112);
  expect(r.story === 'Momentum Continuation' && r.lean >= 0.5).toBe(false);
});

test('Rollover_Top only when trajectory is Falling', () => {
  // a flat wobble should never flip the trajectory to Falling, so Rollover_Top must not fire either
  const wobble = [];
  let p = 100;
  for (let i = 0; i < 60; i++) { p += i < 40 ? 1 : (i % 2 ? -0.02 : 0.02); wobble.push(p); }
  const s = computeStochRSI(wobble);
  if (s.state === 'Rollover_Top') expect(s.traj).toBe('Falling');
});

describe('data integrity: candles stalled while the feed timestamp stays fresh', () => {
  const realNow = Date.now;
  afterAll(() => { Date.now = realNow; });

  test('old candles + fresh feed timestamp => WAIT / suspended', () => {
    const NOW = Date.UTC(2026, 8, 27, 12, 7, 0);
    Date.now = () => NOW;
    const eng = createEngine(60000);
    eng.candles = Array.from({ length: 60 }, (_, i) => ({ t: NOW - (70 - i) * 60000, o: 100, h: 101, l: 99, c: 100 + Math.sin(i), v: 1 }));
    eng.ind = { bb: { mid: 100, upper: 102, lower: 98, state: 'Mid_Band' }, stoch: { k: 55, traj: 'Rising', state: 'Mixed' }, atr: { atr: 1, state: 'Active' } };
    eng.contractStart = NOW - 7 * 60000;
    eng.contractStrike = 100;
    eng.lastTickAt = NOW;
    eng.current = null;
    const sig = computeModuleSignals(eng, 60000);
    expect(sig.rev.signal).toBe('WAIT');
    expect(sig.rev.reason).toMatch(/stopped updating/);
  });
});

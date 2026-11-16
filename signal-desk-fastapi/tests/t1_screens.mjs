import { test, expect, describe, afterAll } from 'vitest';
import { evaluateLeanAndStory, bookModel } from '../src/lib/strategy.js';
import { computeATR } from '../src/lib/indicators.js';

// ---- 1) Story classifier vs the 8 indicator readouts in your two screenshots ----
// bb "Inside" in every screenshot reading; we test with price in the lower half / upper half of the tunnel
const bb = { mid: 100, upper: 110, lower: 90, state: 'Mid_Band' };
const cases = [
  // [label, k, traj, price, expected story, expected lean sign]
  ['BTC  Stoch 17 rising  (revert ↑)', 17, 'Rising', 97, 'Mean Reversion', +1],
  ['ETH  Stoch 24 rising  (revert ↑)', 24, 'Rising', 97, 'Mean Reversion', +1],
  ['SOL  Stoch  3 rising  (revert ↑)',  3, 'Rising', 97, 'Mean Reversion', +1],
  ['XRP  Stoch  1 rising  (revert ↑)',  1, 'Rising', 97, 'Mean Reversion', +1],
  ['SOL  Stoch 19 rising  (revert ↑)', 19, 'Rising', 100, 'Mean Reversion', +1],
  ['BTC1 Stoch  7 falling (mom ↓)',     7, 'Falling', 97, 'Momentum Continuation', -1],
  ['ETH2 Stoch  0 falling (mom ↓)',     0, 'Falling', 100, 'Momentum Continuation', -1],
  ['XRP2 Stoch  0 rising  (revert ↑)',  0, 'Rising', 100, 'Mean Reversion', +1],
];

describe('story classifier vs. reference screenshots', () => {
  for (const [label, k, traj, price, story, sign] of cases) {
    test(label, () => {
      const r = evaluateLeanAndStory({ bb, stoch: { k, traj, state: 'Mixed' } }, price);
      expect(r.story).toBe(story);
      expect(Math.sign(r.lean)).toBe(sign);
    });
  }

  test('outside the top + meter rising = Momentum Continuation, drops snapback', () => {
    const r = evaluateLeanAndStory({ bb, stoch: { k: 70, traj: 'Rising', state: 'High_Rising' } }, 112);
    expect(r.story).toBe('Momentum Continuation');
    expect(r.lean).toBeGreaterThan(0.5);
  });

  test('very high meter turning down near top = Mean Reversion down', () => {
    const r = evaluateLeanAndStory({ bb, stoch: { k: 85, traj: 'Falling', state: 'Rollover_Top' } }, 108);
    expect(r.story).toBe('Mean Reversion');
    expect(r.lean).toBeLessThan(-0.5);
  });

  test('flat meter mid-tunnel = no clear story (wait)', () => {
    const r = evaluateLeanAndStory({ bb, stoch: { k: 50, traj: 'Flat', state: 'Mixed' } }, 100);
    expect(r.story).toBe('Mixed / None');
    expect(Math.abs(r.lean)).toBeLessThan(0.1);
  });

  test('meter falling but price at the bottom edge (indicators fight) = Mixed', () => {
    const r = evaluateLeanAndStory({ bb, stoch: { k: 30, traj: 'Falling', state: 'Mixed' } }, 91);
    expect(r.story).toBe('Mixed / None');
  });
});

// ---- 2) Modeled book vs the four Up quotes in screenshot 2 (3:59 left, 1m candles) ----
describe('modeled book vs. reference screenshot (within ~6¢, one shared calibration constant)', () => {
  const remainingMs = (3 * 60 + 59) * 1000;
  const shots = [
    ['BTC', 78759.64, 78876.78, 16.26, 1.5],
    ['ETH', 2470.82, 2474.31, 0.5920, 2.2],
    ['SOL', 103.320, 103.429, 0.0330, 8.8],
    ['XRP', 1.3812, 1.3838, 0.00047, 6.2],
  ];
  for (const [sym, spot, strike, atr, refBid] of shots) {
    test(`${sym} modeled Up bid vs screenshot ${refBid}¢`, () => {
      const eng = { ind: { atr: { atr } }, contractStrike: strike };
      const b = bookModel(eng, spot, remainingMs, 60000);
      expect(Math.abs(b.up.bid - refBid)).toBeLessThanOrEqual(6);
    });
  }
});

// ---- 3) ATR "asleep" floor is scaled to candle size, not judged against a flat 1-minute floor ----
describe('ATR sleep filter scales with candle size', () => {
  const mk = (n, range, price = 100) =>
    Array.from({ length: n }, (_, i) => ({ t: i, o: price, h: price + range / 2, l: price - range / 2, c: price }));

  test('10s candle with 6bps range is Active (floor scaled by sqrt(10/60))', () => {
    const r = computeATR(mk(30, 0.06), 10000);
    expect(r.state).toBe('Active');
  });
  test('same 6bps range on a 1m candle is Quiet/Asleep (7bps/min floor)', () => {
    const r = computeATR(mk(30, 0.06), 60000);
    expect(r.state).toBe('Quiet/Asleep');
  });
});

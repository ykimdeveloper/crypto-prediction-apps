// Runs the ORIGINAL browser engine on a deterministic candle series and prints JSON for tests/test_parity.py.
// usage (from the project root): node --import ./tests/register.mjs backend/tests/parity_js.mjs
import { createEngine, setEngineCandles, maybeRollContract, refreshStrike } from '../../src/lib/engine.js';
import { computeModuleSignals } from '../../src/lib/strategy.js';

let seed = 12345;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const T0 = Date.UTC(2026, 8, 28, 12, 0, 0);
const N = 140;
let p = 100;
const candles = [];
for (let i = 0; i < N; i++) {
  const o = p;
  // regimes: drift up, drift down, chop -> exercises revert / momentum / mixed stories
  const drift = i < 45 ? 0.05 : i < 90 ? -0.07 : 0.0;
  p = o + drift + (rnd() - 0.5) * 0.5;
  candles.push({ t: T0 + i * 60000, o, h: Math.max(o, p) + rnd() * 0.15, l: Math.min(o, p) - rnd() * 0.15, c: p, v: 1 + rnd() });
}

const cases = [];
const realNow = Date.now;
for (const n of [40, 60, 80, 100, 118, 125, 131, 136, 140]) {
  const closed = candles.slice(0, n);
  const forming = { ...candles[n - 1], t: candles[n - 1].t + 60000, o: candles[n - 1].c, c: candles[n - 1].c + 0.03, h: candles[n - 1].c + 0.1, l: candles[n - 1].c - 0.1 };
  for (const secIntoMinute of [5, 40]) {
    const now = forming.t + secIntoMinute * 1000;
    Date.now = () => now;
    const eng = createEngine(60000);
    setEngineCandles(eng, closed);
    eng.current = forming;
    eng.lastTickAt = now;
    maybeRollContract(eng, now);
    refreshStrike(eng);
    const sig = computeModuleSignals(eng, 60000);
    cases.push({ n, now, ind: eng.ind, strike: eng.contractStrike, contractStart: eng.contractStart, sig });
  }
}
Date.now = realNow;
console.log(JSON.stringify({ candles, forming: null, cases }));

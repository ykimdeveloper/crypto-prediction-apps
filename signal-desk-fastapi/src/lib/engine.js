import { MAX_CANDLES, CONTRACT_MS, MIN_MS } from './constants';
import { computeBollinger, computeStochRSI, computeATR } from './indicators';
// strategy.js imports livePrice from this file; both only use each other's exports inside functions, so the cycle is safe.
import { computeModuleSignals } from './strategy';

export function createEngine(tfMs = MIN_MS) {
  return {
    tfMs,
    candles: [],
    current: null,
    lastTickAt: 0,
    contractStart: null,
    contractStrike: null,
    ind: { bb: null, stoch: null, atr: null },
    lastLog: { rev: null, mom: null, end: null },
    signals: null,
  };
}

export function createEngines(symbols, timeframes) {
  const engines = {};
  symbols.forEach((sym) => {
    engines[sym] = {};
    timeframes.forEach((tf) => {
      engines[sym][tf.key] = createEngine(tf.ms);
    });
  });
  return engines;
}

export function allCloses(eng) {
  return eng.candles.map((c) => c.c);
}

export function livePrice(eng) {
  if (eng.current) return eng.current.c;
  if (eng.candles.length) return eng.candles[eng.candles.length - 1].c;
  return null;
}

export function recomputeIndicators(eng) {
  const closes = allCloses(eng);
  eng.ind.bb = computeBollinger(closes);
  eng.ind.stoch = computeStochRSI(closes);
  eng.ind.atr = computeATR(eng.candles, eng.tfMs);
}

export function setEngineCandles(eng, arr) {
  eng.candles = arr.slice(-MAX_CANDLES);
  recomputeIndicators(eng);
}

// ---------- historical seed aggregation ----------
export function aggregateFromSeconds(secCandles, tfMs) {
  const map = new Map();
  for (const c of secCandles) {
    const bucket = Math.floor(c.t / tfMs) * tfMs;
    let agg = map.get(bucket);
    if (!agg) {
      agg = { t: bucket, o: c.o, h: c.h, l: c.l, c: c.c, v: c.v };
      map.set(bucket, agg);
    } else {
      agg.h = Math.max(agg.h, c.h);
      agg.l = Math.min(agg.l, c.l);
      agg.c = c.c;
      agg.v += c.v;
    }
  }
  return Array.from(map.values()).sort((a, b) => a.t - b.t);
}

// ---------- 15-minute clock-aligned contract ----------
// The strike is the price the contract OPENED at, i.e. the open of the candle that starts at the window boundary.
// (Old code used "whatever the price was when the first message arrived", which is wrong when the app starts mid-window.)
export function strikeAt(eng, winStart) {
  if (eng.current && eng.current.t === winStart) return eng.current.o;
  for (let i = eng.candles.length - 1; i >= 0 && i >= eng.candles.length - 1500; i--) {
    if (eng.candles[i].t === winStart) return eng.candles[i].o;
    if (eng.candles[i].t < winStart) break;
  }
  return eng.current ? eng.current.c : null;
}

export function maybeRollContract(eng, ts) {
  const winStart = Math.floor(ts / CONTRACT_MS) * CONTRACT_MS;
  if (eng.contractStart !== winStart) {
    eng.contractStart = winStart;
    eng.contractStrike = strikeAt(eng, winStart);
  }
}

// after a (re)seed, the exact window-open candle may only now be available
export function refreshStrike(eng) {
  if (eng.contractStart == null) return;
  const s = strikeAt(eng, eng.contractStart);
  if (s != null) eng.contractStrike = s;
}

// ---------- applying native kline_1s / kline_1m updates ----------
export function updateEngineFromKline(eng, k) {
  eng.current = { t: k.t, o: +k.o, h: +k.h, l: +k.l, c: +k.c, v: +k.v };
  eng.lastTickAt = Date.now();
  maybeRollContract(eng, k.t);
  if (k.x) {
    // candle closed on the exchange side
    eng.candles.push(eng.current);
    if (eng.candles.length > MAX_CANDLES) eng.candles.shift();
    recomputeIndicators(eng);
    eng.current = null; // next kline message starts the following bucket
  }
}

// ---------- rolling a finalized 1s candle into a 5s/10s/30s aggregate ----------
export function feedClosedIntoAggregate(eng, closed1s, tfMs) {
  const { o, h, l, c, v, t } = closed1s;
  const bucket = Math.floor(t / tfMs) * tfMs;
  if (!eng.current || eng.current.t !== bucket) {
    if (eng.current) {
      eng.candles.push(eng.current);
      if (eng.candles.length > MAX_CANDLES) eng.candles.shift();
      recomputeIndicators(eng);
    }
    eng.current = { t: bucket, o, h, l, c, v };
  } else {
    eng.current.h = Math.max(eng.current.h, h);
    eng.current.l = Math.min(eng.current.l, l);
    eng.current.c = c;
    eng.current.v += v;
  }
  eng.lastTickAt = Date.now();
  maybeRollContract(eng, t);
}

// ---------- evaluate the three strategies and report which ones just fired ----------
export function evaluateAndLog(eng, tfMs) {
  const sig = computeModuleSignals(eng, tfMs);
  eng.signals = sig;
  const entries = [];
  ['rev', 'mom', 'end'].forEach((k) => {
    const s = sig[k].signal;
    if (s !== 'WAIT' && eng.lastLog[k] !== s) {
      entries.push({ stratKey: k, signal: s, conf: sig[k].conf });
    }
    eng.lastLog[k] = s === 'WAIT' ? null : s;
  });
  return { sig, entries };
}

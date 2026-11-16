import { BB_PERIOD, STOCH_PERIOD, RSI_PERIOD, ATR_PERIOD, ATR_FLOOR_BPS_PER_MIN, MIN_MS } from './constants';

export function sma(arr, period) {
  if (arr.length < period) return null;
  const s = arr.slice(-period);
  return s.reduce((a, b) => a + b, 0) / period;
}

export function stddev(arr, period, mean) {
  const s = arr.slice(-period);
  const v = s.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
  return Math.sqrt(v);
}

export function computeBollinger(closes) {
  if (closes.length < BB_PERIOD) return null;
  const mid = sma(closes, BB_PERIOD);
  const sd = stddev(closes, BB_PERIOD, mid);
  const upper = mid + 2 * sd;
  const lower = mid - 2 * sd;
  const price = closes[closes.length - 1];
  let state = 'Mid_Band';
  if (price >= upper) state = 'Upper_Stretched';
  else if (price <= lower) state = 'Lower_Stretched';
  return { mid, upper, lower, state };
}

export function computeRSISeries(closes, period) {
  if (closes.length < period + 1) return [];
  let gains = 0;
  let losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  const out = [];
  out.push(100 - 100 / (1 + (avgLoss === 0 ? 100 : avgGain / avgLoss)));
  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const gain = diff > 0 ? diff : 0;
    const loss = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    out.push(100 - 100 / (1 + rs));
  }
  return out;
}

export function computeStochRSI(closes) {
  const rsiSeries = computeRSISeries(closes, RSI_PERIOD);
  if (rsiSeries.length < STOCH_PERIOD) return null;
  const kRaw = [];
  for (let i = STOCH_PERIOD - 1; i < rsiSeries.length; i++) {
    const win = rsiSeries.slice(i - STOCH_PERIOD + 1, i + 1);
    const hi = Math.max(...win);
    const lo = Math.min(...win);
    kRaw.push(hi === lo ? 50 : ((rsiSeries[i] - lo) / (hi - lo)) * 100);
  }
  const kSmooth = [];
  for (let i = 2; i < kRaw.length; i++) kSmooth.push((kRaw[i] + kRaw[i - 1] + kRaw[i - 2]) / 3);
  if (kSmooth.length < 3) return null;
  const k = kSmooth[kSmooth.length - 1];
  const prevK = kSmooth[kSmooth.length - 2];
  let traj = 'Flat';
  const delta = k - prevK;
  if (delta > 0.5) traj = 'Rising';
  else if (delta < -0.5) traj = 'Falling';
  let state = 'Mixed';
  if (k > 60 && traj === 'Rising') state = 'High_Rising';
  else if (k < 40 && traj === 'Falling') state = 'Low_Falling';
  if (k > 70 && delta < 0) state = 'Rollover_Top';
  if (k < 30 && delta > 0) state = 'Bounce_Bottom';
  return { k, prevK, delta, traj, state };
}

export function computeATR(candles, tfMs = MIN_MS) {
  if (candles.length < ATR_PERIOD + 1) return null;
  const trs = [];
  for (let i = 1; i < candles.length; i++) {
    const c = candles[i];
    const p = candles[i - 1];
    trs.push(Math.max(c.h - c.l, Math.abs(c.h - p.c), Math.abs(c.l - p.c)));
  }
  const atr = sma(trs, ATR_PERIOD);
  const price = candles[candles.length - 1].c;
  // floor is 7 bps of price per MINUTE; scale by sqrt(candle length / 1 minute) so 1s/10s candles aren't judged against a 1m floor
  const floor = price * ATR_FLOOR_BPS_PER_MIN * Math.sqrt(tfMs / MIN_MS);
  return { atr, state: atr < floor ? 'Quiet/Asleep' : 'Active', floor };
}

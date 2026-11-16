import { CONTRACT_MS, MIN_MS, TIMING, PRICE, BOOK, EOM_CONF_SIGMA, STALE_MS } from './constants';
import { fmt, clampConf } from './format';
import { livePrice } from './engine';

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const round1 = (v) => Math.round(v * 10) / 10;

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}
export const normalCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

// where the price sits inside the Bollinger tunnel: 0 = middle, -1 = lower band, +1 = upper band (beyond = outside)
export function bbPosition(bb, price) {
  if (!bb) return 0;
  const up = bb.upper - bb.mid || 1e-9;
  const dn = bb.mid - bb.lower || 1e-9;
  return price >= bb.mid ? (price - bb.mid) / up : (price - bb.mid) / dn;
}

// ---------- story / lean classifier (pure) ----------
// Follows the walkthrough: two independent read-outs compete.
//   momentum ("keep going"): StochRSI LEVEL + DIRECTION — mid/high and still rising = up, mid/low and still falling = down.
//   snapback ("bounce/fade"): a very low meter turning up (bounce) or a very high meter turning down (fade),
//                             stronger the closer price is to that edge of the tunnel, plus a small edge-only term.
// Outside the tunnel with the meter agreeing = breakout: momentum wins and snapback is dropped.
// Otherwise the stronger one wins; both weak = no lean; opposite signs and both real = mixed (indicators fighting).
// Check against the reference screenshot: Stoch 17/19/24/3/1/0 rising -> "revert ↑"; Stoch 7/0 falling -> "mom ↓".
export function evaluateLeanAndStory(ind, price) {
  if (!ind.bb || !ind.stoch) return { story: 'Mixed / None', lean: 0, momentum: 0, snap: 0 };
  const { k, traj } = ind.stoch;
  const pos = bbPosition(ind.bb, price);

  let momentum = 0;
  if (traj === 'Rising' && k >= 45) momentum = 0.35 + (0.65 * (k - 45)) / 55;
  else if (traj === 'Falling' && k <= 55) momentum = -(0.35 + (0.65 * (55 - k)) / 55);

  const breakoutConfirmed = (pos >= 1 && momentum > 0) || (pos <= -1 && momentum < 0);
  if (breakoutConfirmed) momentum = clamp(momentum * 1.25, -1, 1);

  let snapTurn = 0;
  if (traj === 'Rising' && k < 30) {
    const turn = (30 - k) / 30;
    const near = clamp((-pos - 0.2) / 0.8, 0, 1);
    snapTurn = (0.4 + 0.6 * turn) * (0.5 + 0.5 * near);
  } else if (traj === 'Falling' && k > 70) {
    const turn = (k - 70) / 30;
    const near = clamp((pos - 0.2) / 0.8, 0, 1);
    snapTurn = -(0.4 + 0.6 * turn) * (0.5 + 0.5 * near);
  }
  let snapEdge = 0;
  if (pos <= -0.85) snapEdge = 0.35 * clamp(0.5 + (-pos - 0.85) / 0.3, 0.5, 1);
  else if (pos >= 0.85) snapEdge = -0.35 * clamp(0.5 + (pos - 0.85) / 0.3, 0.5, 1);
  const snap = clamp(snapTurn + snapEdge, -1, 1);

  const bothWeak = Math.abs(momentum) < 0.15 && Math.abs(snap) < 0.15;
  const fighting = !bothWeak && Math.abs(momentum) >= 0.15 && Math.abs(snap) >= 0.15 && Math.sign(momentum) !== Math.sign(snap);

  let story;
  let lean;
  if (breakoutConfirmed) { story = 'Momentum Continuation'; lean = momentum; }
  else if (bothWeak) { story = 'Mixed / None'; lean = ((momentum + snap) / 2) * 0.4; }
  else if (fighting) { story = 'Mixed / None'; lean = ((momentum + snap) / 2) * 0.5; }
  else if (Math.abs(snap) > Math.abs(momentum)) { story = 'Mean Reversion'; lean = snap; }
  else { story = 'Momentum Continuation'; lean = momentum; }
  return { story, lean: clamp(lean, -1, 1), momentum, snap };
}

// ---------- modeled contract book (cents, 0-100) ----------
// P(finish above strike) = Phi(distance / (ATR * sqrt(candles left) * fudge)). See BOOK in constants.js for the calibration.
export function bookModel(eng, price, remainingMs, tfMs) {
  if (!eng.ind.atr || eng.contractStrike == null || price == null) return null;
  const candlesLeft = Math.max(remainingMs / tfMs, 0.02);
  const sigma = (eng.ind.atr.atr || 1e-12) * Math.sqrt(candlesLeft) * BOOK.sigmaFudge;
  const pUp = clamp(normalCdf((price - eng.contractStrike) / sigma) * 100, 0.5, 99.5);
  const h = BOOK.halfSpread;
  const upBid = round1(clamp(pUp - h, 0.1, 99.8));
  const upAsk = round1(clamp(pUp + h, 0.2, 99.9));
  return {
    pUp,
    up: { bid: upBid, ask: upAsk },
    down: { bid: round1(100 - upAsk), ask: round1(100 - upBid) },
  };
}

const short = (story) => (story === 'Mean Reversion' ? 'revert' : story === 'Momentum Continuation' ? 'mom' : 'mixed');
const WAIT = (reason, conf = null) => ({ signal: 'WAIT', conf, reason });

// ---------- the three strategies ----------
// Unlike the old version there is NO mutually-exclusive "module A / module B" switch: each strategy has its own clock.
//   MID REV / MID MOM: entries only while >= 5:00 remains (the reference shows "Under five minutes"), take-profit watched to 1:00.
//   END-MKT:           only between 8:00 and 2:00 remaining, so it overlaps the mid-market forks from 8:00 to 5:00.
export function computeModuleSignals(eng, tfMs) {
  const price = livePrice(eng);
  const warming = !eng.ind.bb || !eng.ind.stoch || !eng.ind.atr || eng.contractStart == null || price == null;
  const stale = eng.lastTickAt && Date.now() - eng.lastTickAt > STALE_MS;
  if (warming || stale) {
    const why = stale ? 'Feed stale for this window.' : 'Warming up — not enough closed candles yet.';
    return { module: '—', story: null, lean: 0, book: null, remMin: null, rev: WAIT(why), mom: WAIT(why), end: WAIT(why) };
  }

  const { story, lean } = evaluateLeanAndStory(eng.ind, price);
  const remainingMs = eng.contractStart + CONTRACT_MS - Date.now();
  const remMin = remainingMs / MIN_MS;
  const strike = eng.contractStrike;
  const book = bookModel(eng, price, remainingMs, tfMs);
  const sleepOk = eng.ind.atr.state !== 'Quiet/Asleep';
  const dir = lean > 0 ? 'UP' : lean < 0 ? 'DOWN' : null;
  const sideAsk = (d) => (d === 'UP' ? book.up.ask : book.down.ask);

  const midFork = (kind) => {
    const want = kind === 'rev' ? 'Mean Reversion' : 'Momentum Continuation';
    if (story !== want) return WAIT(`Not ${kind === 'rev' ? 'rev' : 'mom'} — story is ${short(story)}.`);
    if (remMin < TIMING.midEntryMinRemaining) return WAIT('Under five minutes.');
    if (!sleepOk) return WAIT('ATR asleep — sleep filter blocks the mid-market forks.');
    if (!dir) return WAIT('No lean.');
    const { k, traj, state } = eng.ind.stoch;
    let stochOk;
    if (kind === 'rev') {
      stochOk = dir === 'UP' ? state === 'Bounce_Bottom' || (k <= 40 && traj === 'Rising')
        : state === 'Rollover_Top' || (k >= 60 && traj === 'Falling');
    } else {
      stochOk = dir === 'UP' ? k >= 50 && k <= 85 && traj === 'Rising' : k <= 50 && k >= 15 && traj === 'Falling';
    }
    const conf = clampConf(50 + Math.abs(lean) * 38 + 8);
    if (!stochOk) return WAIT('Stoch not confirmed.', conf);
    const ask = sideAsk(dir);
    const [lo, hi] = kind === 'rev' ? PRICE.mmrBand : PRICE.mmcBand;
    if (ask < PRICE.minEntry) return WAIT(`${dir} is ${ask}¢ — under the ${PRICE.minEntry}¢ minimum.`, conf);
    if (ask < lo) return WAIT(`${dir} is ${ask}¢ — below the ${lo}–${hi}¢ entry band.`, conf);
    if (ask > hi) return WAIT(`${dir} is ${ask}¢ — above the ${lo}–${hi}¢ entry band.`, conf);
    const tp = kind === 'rev' ? PRICE.tpMmr : PRICE.tpMmc;
    return {
      signal: `BUY ${dir}`, conf, ask, tp,
      reason: kind === 'rev'
        ? `Snapback: ${dir} at ${ask}¢, target +${tp}¢, no stop.`
        : `Breakout running: ${dir} at ${ask}¢, target +${tp}¢, no stop.`,
    };
  };

  const rev = midFork('rev');
  const mom = midFork('mom');

  // ---- End-of-Market ----
  // Before 8:00 the box still shows which side price is on and how confident the model is (the reference bot shows
  // "lean down 87%" at 11 minutes left) but it stays WAIT: no rec, no paper trade, no alert until 8:00.
  let end;
  if (remMin < TIMING.eomMinRemaining) {
    end = WAIT(`Closed under ${TIMING.eomMinRemaining}:00 remaining.`);
  } else {
    const favorUp = price >= strike;
    const eDir = favorUp ? 'UP' : 'DOWN';
    const ask = sideAsk(eDir);
    const candlesLeft = Math.max(remainingMs / tfMs, 0.02);
    const cushion = Math.abs(price - strike);
    const buffer = eng.ind.atr.atr * Math.sqrt(candlesLeft);
    // indicators are only "light seasoning" here (<= 20% of the confidence)
    const statConf = normalCdf(cushion / (buffer * EOM_CONF_SIGMA || 1e-12)) * 100;
    const agree = eDir === 'UP' ? lean : -lean;
    const conf = clamp(Math.round(0.8 * statConf + 0.2 * (50 + 50 * agree)), 1, 99);
    if (remMin > TIMING.eomMaxRemaining) {
      end = WAIT(`Leaning ${eDir} (${ask}¢) — buys open at ${TIMING.eomMaxRemaining}:00 remaining.`, conf);
    } else if (ask > PRICE.eomBand[1]) {
      end = { signal: `HOLD ${eDir}`, conf, reason: `above ${PRICE.eomBand[1]}¢ — nothing left to earn.` };
    } else if (ask < PRICE.eomBand[0]) {
      end = WAIT(`${eDir} is ${ask}¢ — below ${PRICE.eomBand[0]}¢, cushion ${fmt(cushion)} vs buffer ${fmt(buffer)}.`, conf);
    } else {
      end = {
        signal: `BUY ${eDir}`, conf, ask, tp: null,
        reason: `${PRICE.eomBand[0]}–${PRICE.eomBand[1]}¢ · cushion ${fmt(cushion)} vs buffer ${fmt(buffer)}. Holds to settlement.`,
      };
    }
  }

  const overlap = remMin > TIMING.midEntryMinRemaining && remMin <= TIMING.eomMaxRemaining;
  const module = remMin > TIMING.eomMaxRemaining ? 'MID' : overlap ? 'MID+END' : remMin >= TIMING.eomMinRemaining ? 'END' : 'CLOSING';
  return { module, story, lean, book, remMin, rev, mom, end };
}

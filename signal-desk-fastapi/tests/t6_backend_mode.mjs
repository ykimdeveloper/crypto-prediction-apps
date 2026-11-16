import { test, expect, describe } from 'vitest';
import { EXCHANGES, ccxtExchange, classifyFetchError } from '../src/lib/exchanges.js';
import { serverCard, emptyCard, EMPTY_PAPER } from '../src/lib/serverCards.js';
import { wsUrl, MODES, resolveSource } from '../src/lib/backend.js';

const byId = (id) => EXCHANGES.find((e) => e.id === id);

describe('CCXT data source', () => {
  test('wraps an exchange so the 1-minute polling path can use it unchanged', () => {
    const ex = ccxtExchange(byId('kraken'));
    expect(ex.multi).toBe(false);
    expect(ex.ccxt).toBe(true);
    expect(ex.pair('BTC')).toBe('BTC/USD');
    expect(ex.url(ex.pair('BTC'), 100)).toBe('/api/candles?exchange=kraken&symbol=BTC%2FUSD&limit=100');
  });

  test('id mapping: Binance Global -> binance, Coinbase -> coinbaseexchange', () => {
    expect(ccxtExchange(byId('binance2')).ccxtId).toBe('binance');
    expect(ccxtExchange(byId('coinbase')).ccxtId).toBe('coinbaseexchange');
    expect(ccxtExchange(byId('okx')).ccxtId).toBe('okx');
  });

  test('every exchange in the list can be wrapped', () => {
    EXCHANGES.forEach((e) => expect(ccxtExchange(e).url('X/Y', 5)).toMatch(/^\/api\/candles\?exchange=/));
  });

  test('parse maps the backend payload to {t,o,h,l,c,v} and rejects error bodies', () => {
    const ex = ccxtExchange(byId('okx'));
    const out = ex.parse({ candles: [{ t: 60000, o: 1, h: 2, l: 0.5, c: 1.5, v: 9 }] });
    expect(out).toEqual([{ t: 60000, o: 1, h: 2, l: 0.5, c: 1.5, v: 9 }]);
    expect(() => ex.parse({ detail: 'nope' })).toThrow('unexpected response');
  });

  test('error notes talk about the backend, not CORS', () => {
    const ex = ccxtExchange(byId('okx'));
    expect(classifyFetchError(new TypeError('Failed to fetch'), ex)).toMatch(/FastAPI backend/);
    expect(classifyFetchError(new Error('HTTP 404'), ex)).toMatch(/not listed/);
    expect(classifyFetchError(new Error('HTTP 502'), ex)).toMatch(/CCXT/);
    // original exchanges keep the original wording
    expect(classifyFetchError(new TypeError('Failed to fetch'), byId('okx'))).toMatch(/CORS/);
  });

  test('modes', () => {
    expect(MODES.map((m) => m.id)).toEqual(['custom', 'ccxt']); // FastAPI is no longer a data route: it is the separate ML toggle
    expect(wsUrl('/ws/desk')).toMatch(/^ws:\/\/.*\/ws\/desk$/);
  });
});

describe('ML · FastAPI toggle is independent of the browser route', () => {
  test('fresh install: browser engine, custom routes', () => {
    expect(resolveSource('custom', null)).toEqual({ mode: 'custom', ml: false });
    expect(resolveSource(undefined, undefined)).toEqual({ mode: 'custom', ml: false });
  });
  test('an explicit saved ml flag wins, and keeps the browser route underneath', () => {
    expect(resolveSource('ccxt', true)).toEqual({ mode: 'ccxt', ml: true });
    expect(resolveSource('custom', false)).toEqual({ mode: 'custom', ml: false });
  });
  test('old three-way switch: fastapi and ccxt both meant "server does the work"', () => {
    expect(resolveSource('fastapi', null)).toEqual({ mode: 'custom', ml: true });
    expect(resolveSource('ccxt', null)).toEqual({ mode: 'ccxt', ml: true });
  });
  test('garbage falls back to the browser engine', () => {
    expect(resolveSource('nope', 'yes')).toEqual({ mode: 'custom', ml: false });
  });
});

describe('FastAPI card mapping', () => {
  const NOW = 1_000_000 * 900_000 + 5 * 60_000; // 5:00 into a window
  const snap = {
    symbol: 'BTC', failed: false, price: 100.5, strike: 100, contractStart: NOW - 5 * 60_000, mlPUp: 0.574,
    spotBook: { bid: 100.49, ask: 100.51 },
    ind: { bb: { mid: 100, upper: 101, lower: 99, state: 'Upper_Stretched' }, stoch: { k: 72.4, traj: 'Rising', state: 'High_Rising' }, atr: { atr: 0.31, state: 'Quiet/Asleep' } },
    sig: { module: 'MID+END', story: 'Momentum Continuation', lean: 0.62, book: { pUp: 70, up: { bid: 69.5, ask: 70.5 }, down: { bid: 29.5, ask: 30.5 } },
      rev: { signal: 'WAIT', conf: null, reason: 'x' }, mom: { signal: 'BUY UP', conf: 82, ask: 70.5, tp: 20, reason: 'y' }, end: { signal: 'WAIT', conf: 60, reason: 'z' } },
  };

  test('formats like the browser engine does', () => {
    const c = serverCard(snap, 'USDT', 'Binance · CCXT', 'BTC/USDT', NOW);
    expect(c.spot).toBe('100.50');
    expect(c.strike).toBe('100.00');
    expect(c.dist).toBe('0.5000 above strike');
    expect(c.stoch).toBe('72 ↑');
    expect(c.bb).toBe('Upper Stretched');
    expect(c.atr).toBe('0.3100 (asleep) · mom ↑');
    expect(c.lean).toBe('+0.62');
    expect(c.atrState).toBe('Asleep');
    expect(c.closesIn).toBe('10:00');
    expect(c.spotBook).toBe('100.49 / 100.51');
    expect(c.ml).toBe('57%');
    expect(c.mom.signal).toBe('BUY UP');
  });

  test('before the first evaluation the card is the loading placeholder', () => {
    const c = serverCard({ symbol: 'ETH', failed: false, ind: null, sig: null }, 'USDT', 'x', 'ETH/USDT', NOW);
    expect(c).toEqual(emptyCard('ETH', 'USDT'));
  });

  test('failed feed says so', () => {
    const c = serverCard({ symbol: 'ZZZ', failed: true, ind: null, sig: null }, 'USDT', 'Binance', 'ZZZ/USDT', NOW);
    expect(c.failed).toBe(true);
    expect(c.dist).toMatch(/may not be listed/);
  });

  test('empty paper stats have the shape PaperPanel reads', () => {
    ['rev', 'mom', 'end'].forEach((k) => {
      expect(EMPTY_PAPER[k].gate.hold.n).toBe(0);
      expect(EMPTY_PAPER[k].rows).toEqual([]);
    });
  });
});

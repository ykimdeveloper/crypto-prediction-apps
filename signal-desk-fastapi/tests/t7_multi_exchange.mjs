import { test, expect, describe } from 'vitest';
import { ccxtChoices, aggExchange, DEFAULT_CCXT_EXCHANGES, USD_LIKE } from '../src/lib/exchanges.js';
import { serverCard } from '../src/lib/serverCards.js';

describe('CCXT multi-exchange selection', () => {
  test('choices are CCXT ids, dollar-quoted only, no duplicates', () => {
    const ids = ccxtChoices().map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toContain('coinbaseexchange');
    expect(ids).not.toContain('upbit');   // KRW
    expect(ids).not.toContain('coinjar'); // AUD
    expect(ids).not.toContain('binance2');
    DEFAULT_CCXT_EXCHANGES.forEach((d) => expect(ids).toContain(d));
    expect(USD_LIKE).toEqual(['USD', 'USDT', 'USDC']);
  });

  test('aggExchange describes the selection', () => {
    const ex = aggExchange(['kraken', 'okx']);
    expect(ex).toMatchObject({ agg: true, multi: false, quote: 'composite', name: '2 exchanges' });
    expect(aggExchange(['kraken']).name).toBe('Kraken');
  });

  test('cards carry per-exchange chips and a readable no-data message', () => {
    const v = [{ exchange: 'kraken', status: 'live', symbol: 'BTC/USD', price: 100.4, devBps: -10 }];
    const ok = serverCard({ symbol: 'BTC', price: 100.5, strike: 100, contractStart: 0, venues: v,
      ind: { bb: { state: 'Mid_Band' }, stoch: { k: 50, traj: 'Flat' }, atr: { atr: 1, state: 'Active' } },
      sig: { module: 'MID', story: null, lean: 0, book: null, rev: {}, mom: {}, end: {} } }, 'composite', '2 exchanges', 'BTC');
    expect(ok.venues[0]).toMatchObject({ exchange: 'kraken', price: '100.40', devBps: '-10.0bp' });
    const bad = serverCard({ symbol: 'ZZZ', failed: true, error: 'no selected exchange has it', venues: [], ind: null, sig: null }, 'composite', 'x', 'ZZZ');
    expect(bad.dist).toMatch(/no selected exchange has it/);
  });
});

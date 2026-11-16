import { test, expect } from 'vitest';
import { shouldAlert, DEFAULT_ALERTS } from '../src/lib/alerts.js';

const on = { ...DEFAULT_ALERTS, enabled: true };
const rec = { sym: 'BTC', stratKey: 'end', signal: 'BUY UP' };

test('off by default, nothing alerts', () => {
  expect(shouldAlert(DEFAULT_ALERTS, rec)).toBe(false);
});

test('enabled alerts for any ticker and strategy', () => {
  expect(shouldAlert(on, rec)).toBe(true);
});

test('muted ticker is skipped, others still alert', () => {
  const muted = { ...on, muted: ['BTC'] };
  expect(shouldAlert(muted, rec)).toBe(false);
  expect(shouldAlert(muted, { ...rec, sym: 'ETH' })).toBe(true);
});

test('strategy toggle off is skipped', () => {
  expect(shouldAlert({ ...on, strats: { ...on.strats, end: false } }, rec)).toBe(false);
});

test('a ticker added later is not muted by default', () => {
  expect(shouldAlert({ ...on, muted: ['BTC'] }, { ...rec, sym: 'DOGE' })).toBe(true);
});

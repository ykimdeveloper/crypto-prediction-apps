export function symShort(s) {
  return s.replace('USDT', '');
}

export function fmt(price) {
  if (price == null || isNaN(price)) return '—';
  const d = price < 0.01 ? 8 : price < 0.1 ? 6 : price < 5 ? 4 : price < 100 ? 3 : 2;
  return price.toFixed(d);
}

export function fmtRemaining(ms) {
  if (ms == null) return '—';
  ms = Math.max(0, ms);
  if (ms < 60000) return (ms / 1000).toFixed(1) + 's';
  const totalSec = Math.floor(ms / 1000);
  const mm = Math.floor(totalSec / 60);
  const ss = totalSec % 60;
  return mm + ':' + String(ss).padStart(2, '0');
}

export function clampConf(v) {
  return Math.max(50, Math.min(97, Math.round(v)));
}

export function fmtCents(c) {
  if (c == null || isNaN(c)) return '—';
  return (c >= 0 ? '+' : '−') + Math.abs(c).toFixed(1) + '¢';
}

export function fmtUsd(cents) {
  if (cents == null || isNaN(cents)) return '—';
  const d = cents / 100;
  return (d < 0 ? '−$' : '$') + Math.abs(d).toFixed(2);
}

// Where the FastAPI backend lives, and the two independent header controls that decide how data is produced.
//
//   Browser route (MODES)   only applies while ML is OFF — everything is computed in the browser:
//     custom  the original per-exchange REST/WebSocket routes
//     ccxt    candles come from FastAPI (CCXT Pro on the server); indicators + strategy still run in the browser
//   ML · FastAPI (boolean)  ON = the server does everything (any CCXT exchange, ta indicators, strategies, paper trading,
//                           SQLite, the h2o ML ↑ probability) and streams finished cards over /ws/desk. The browser-route
//                           switch is greyed out, because the server always fetches through CCXT.
export const MODES = [
  { id: 'custom', label: 'Custom routes', hint: 'browser → each exchange directly' },
  { id: 'ccxt', label: 'CCXT', hint: 'candles via FastAPI + CCXT Pro, indicators and strategies still in the browser' },
];

// Turns whatever was saved in localStorage into { mode, ml }. Older builds had one three-way switch:
//   'fastapi' (server, one exchange) and 'ccxt' (server, many exchanges merged) both meant "server does the work" -> ml on.
export function resolveSource(savedMode, savedMl) {
  const mode = savedMode === 'ccxt' ? 'ccxt' : 'custom';
  if (typeof savedMl === 'boolean') return { mode, ml: savedMl };
  return { mode, ml: savedMode === 'fastapi' || savedMode === 'ccxt' };
}

// Dev: leave VITE_API_BASE unset — vite.config.js proxies /api and /ws to 127.0.0.1:8000.
// Production: set VITE_API_BASE=https://your-backend.example.com (or serve both behind one reverse proxy).
const apiBase = () => (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE) || '';

export const apiUrl = (path) => apiBase() + path;

export function wsUrl(path) {
  const b = apiBase();
  if (b) return b.replace(/^http/, 'ws') + path;
  const proto = typeof location !== 'undefined' && location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${typeof location !== 'undefined' ? location.host : 'localhost'}${path}`;
}

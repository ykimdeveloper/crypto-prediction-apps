import { apiUrl } from './backend';

// Every parser returns [{t (ms), o, h, l, c, v}] in any order; the hook sorts by time.
const c = (t, o, h, l, cl, v) => ({ t: +t, o: +o, h: +h, l: +l, c: +cl, v: +v });
// seconds, milliseconds or ISO string -> ms
const ts = (x) => (typeof x === 'string' && Number.isNaN(+x) ? Date.parse(x) : +x < 1e12 ? +x * 1000 : +x);
const arr = (x) => {
  if (!Array.isArray(x)) throw new Error('unexpected response');
  return x;
};

// The old "pair not listed, rate limit, or blocked by CORS" note collapsed three very different problems into one
// unreadable message. This turns whatever poll() caught back into which one it actually was.
//   - a real HTTP status  -> the fetch reached the exchange and it answered (rate limit, bad pair, geofence, 5xx)
//   - "unexpected response" -> parse() got JSON back but not an array — usually an error body for an unlisted pair
//   - a bare TypeError ("Failed to fetch"/"Load failed") -> the browser never got a response at all: CORS or network
export function classifyFetchError(e, ex) {
  const msg = (e && e.message) || String(e);
  const httpMatch = msg.match(/^HTTP (\d+)/);
  if (ex && ex.ccxt) {
    // these statuses come from OUR backend (backend/app/main.py), not from the exchange
    if (httpMatch) {
      const code = +httpMatch[1];
      if (code === 400) return `HTTP 400 — CCXT has no exchange id "${ex.ccxtId}" (edit CCXT_IDS in src/lib/exchanges.js)`;
      if (code === 404) return 'pair not listed on this exchange (per CCXT markets)';
      if (code === 502) return 'the backend could not get candles from the exchange through CCXT (rate limit, geo-block, or exchange down) — see the uvicorn log';
      if (code === 503) return 'backend is still starting this feed (or ccxt is not installed there) — retrying';
      return `HTTP ${code} from the FastAPI backend`;
    }
    if (typeof TypeError !== 'undefined' && e instanceof TypeError) {
      return 'cannot reach the FastAPI backend — is `uvicorn app.main:app --port 8000` running in ./backend?';
    }
  }
  if (httpMatch) {
    const code = +httpMatch[1];
    if (code === 429) return 'rate limited (HTTP 429) — this exchange is throttling requests from this browser/IP';
    if (code === 403 || code === 401) {
      return `HTTP ${code} — access denied by the exchange or its CDN (IP/region block or bot filter), not a parsing problem`;
    }
    if (code === 451) return `HTTP 451 — blocked in this region (exchange geofencing, not a code issue)`;
    if (code === 400 || code === 404) return `HTTP ${code} — pair is probably not listed on this exchange`;
    if (code >= 500) return `HTTP ${code} — exchange-side error, try again shortly`;
    return `HTTP ${code}`;
  }
  if (msg === 'unexpected response') return 'pair not listed — got an error payload instead of a candle array (check the Network tab for the exchange\'s own message)';
  if (typeof TypeError !== 'undefined' && e instanceof TypeError) {
    return 'blocked by CORS or unreachable — the browser never got a response (see Footer: this exchange may need the dev proxy)';
  }
  return msg;
}

// MEXC, Crypto.com, WhiteBIT, BTSE and CoinJar don't (or can't be relied on to) send CORS headers for direct browser requests. In dev, Vite's proxy (see
// vite.config.js) fetches these server-side instead so the browser sees a same-origin response; in a production
// build there's no dev server, so these fall back to the direct URL and will fail with the same CORS error unless
// you add an equivalent proxy route (a serverless function, Nginx rule, etc.) at deploy time.
export const PROXIED_EXCHANGES = [
  'mexc',
  'cryptocom',
  'whitebit',
  'btse',
  'coinjar',
  'kucoin',
  'bitfinex',
  // 'binanceus',
];

const base = (dev, prod) => (import.meta.env.DEV ? dev : prod);

// Binance is the only "multi" exchange: native 1s + 1m klines over WebSocket (see useSignalDesk).
// All others are 1-minute only and are polled over their public REST kline endpoint.
export const EXCHANGES = [
  {
    id: 'binance',
    name: 'Binance',
    multi: true,
    quote: 'USDT',
    pair: (b) => b + 'USDT',
  },
  {
    id: 'mexc',
    name: 'MEXC',
    quote: 'USDT',
    pair: (b) => b + 'USDT',
    url: (p, n) =>
      `${base('/proxy/mexc', 'https://api.mexc.com')}/api/v3/klines?symbol=${p}&interval=1m&limit=${n}`,
    parse: (j) => arr(j).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  {
    id: 'gate',
    name: 'Gate.io',
    quote: 'USDT',
    pair: (b) => b + '_USDT',
    url: (p, n) =>
      `https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair=${p}&interval=1m&limit=${n}`,
    parse: (j) =>
      arr(j).map((r) => c(r[0] * 1000, r[5], r[3], r[4], r[2], r[6])),
  },
  {
    id: 'cryptocom',
    name: 'Crypto.com',
    quote: 'USDT',
    pair: (b) => b + '_USDT',
    pollMs: 4000,
    url: (p) =>
      `${base('/proxy/cryptocom', 'https://api.crypto.com')}/v2/public/get-candlestick?instrument_name=${p}&timeframe=1m`,
    parse: (j) =>
      arr(j.result?.data).map((d) => c(d.t, d.o, d.h, d.l, d.c, d.v)),
  },
  {
    id: 'bitget',
    name: 'Bitget',
    quote: 'USDT',
    pair: (b) => b + 'USDT',
    url: (p, n) =>
      `https://api.bitget.com/api/v2/spot/market/candles?symbol=${p}&granularity=1min&limit=${n}`,
    parse: (j) => arr(j.data).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  {
    id: 'okx',
    name: 'OKX',
    quote: 'USDT',
    pair: (b) => b + '-USDT',
    url: (p, n) =>
      `https://www.okx.com/api/v5/market/candles?instId=${p}&bar=1m&limit=${n}`,
    parse: (j) => arr(j.data).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  {
    id: 'coinbase',
    name: 'Coinbase',
    quote: 'USD',
    pair: (b) => b + '-USD',
    url: (p) =>
      `https://api.exchange.coinbase.com/products/${p}/candles?granularity=60`,
    parse: (j) =>
      arr(j).map((r) => c(r[0] * 1000, r[3], r[2], r[1], r[4], r[5])),
  },
  {
    id: 'htx',
    name: 'HTX (Huobi)',
    quote: 'USDT',
    pair: (b) => b.toLowerCase() + 'usdt',
    url: (p, n) =>
      `https://api.huobi.pro/market/history/kline?symbol=${p}&period=1min&size=${n}`,
    parse: (j) =>
      arr(j.data).map((d) =>
        c(d.id * 1000, d.open, d.high, d.low, d.close, d.amount),
      ),
  },
  {
    id: 'upbit',
    name: 'Upbit',
    quote: 'KRW',
    pair: (b) => 'KRW-' + b,
    url: (p, n) =>
      `https://api.upbit.com/v1/candles/minutes/1?market=${p}&count=${n}`,
    parse: (j) =>
      arr(j).map((d) =>
        c(
          Date.parse(d.candle_date_time_utc + 'Z'),
          d.opening_price,
          d.high_price,
          d.low_price,
          d.trade_price,
          d.candle_acc_trade_volume,
        ),
      ),
  },
  // {
  //   id: 'binanceus',
  //   name: 'Binance.US',
  //   quote: 'USDT',
  //   pair: (b) => b + 'USDT',
  //   url: (p, n) =>
  //     `https://api.binance.us/api/v3/klines?symbol=${p}&interval=1m&limit=${n}`,
  //   parse: (j) => arr(j).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  // },
  {
    id: 'binanceus',
    name: 'Binance.US',
    quote: 'USDT',
    pair: (b) => b + 'USDT',
    url: (p, n) =>
      `${base('/proxy/binanceus', 'https://api.binance.us')}/api/v3/klines?symbol=${encodeURIComponent(p)}&interval=1m&limit=${n}`,
    parse: (j) => arr(j).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  {
    id: 'binance2',
    name: 'Binance Global',
    quote: 'USDT',
    pair: (b) => b + 'USDT',
    url: (p, n) =>
      `/proxy/binance/api/v3/klines?symbol=${encodeURIComponent(p)}&interval=1m&limit=${n}`,
    parse: (j) => arr(j).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  {
    id: 'kraken',
    name: 'Kraken',
    quote: 'USD',
    pollMs: 5000,
    pair: (b) => (({ BTC: 'XBT', DOGE: 'XDG' })[b] || b) + 'USD',
    url: (p) => `https://api.kraken.com/0/public/OHLC?pair=${p}&interval=1`,
    // Kraken answers HTTP 200 with {error:[...]} for bad pairs; result is keyed by Kraken's own pair name (e.g. XXBTZUSD).
    parse: (j) => {
      if (j.error && j.error.length) throw new Error('unexpected response');
      const key = Object.keys(j.result || {}).find((k) => k !== 'last');
      return arr(j.result && j.result[key]).map((r) =>
        c(r[0] * 1000, r[1], r[2], r[3], r[4], r[6]),
      );
    },
  },
  {
    id: 'bitstamp',
    name: 'Bitstamp',
    quote: 'USD',
    pair: (b) => b.toLowerCase() + 'usd',
    url: (p, n) =>
      `https://www.bitstamp.net/api/v2/ohlc/${p}/?step=60&limit=${n}`,
    parse: (j) =>
      arr(j.data?.ohlc).map((d) =>
        c(d.timestamp * 1000, d.open, d.high, d.low, d.close, d.volume),
      ),
  },
  // {
  //   id: 'bitfinex',
  //   name: 'Bitfinex',
  //   quote: 'USD',
  //   pollMs: 6000, // public candle limit is only ~10-30 req/min
  //   pair: (b) => (b.length > 3 ? `t${b}:USD` : `t${b}USD`),
  //   url: (p, n) =>
  //     `https://api-pub.bitfinex.com/v2/candles/trade:1m:${p}/hist?limit=${n}`,
  //   // Bitfinex row order is [mts, open, CLOSE, high, low, volume]
  //   parse: (j) => {
  //     if (Array.isArray(j) && j[0] === 'error')
  //       throw new Error('unexpected response');
  //     return arr(j).map((r) => c(r[0], r[1], r[3], r[4], r[2], r[5]));
  //   },
  // },
  {
    id: 'bitfinex',
    name: 'Bitfinex',
    quote: 'USD',
    pollMs: 6000,
    pair: (b) => (b.length > 3 ? `t${b}:USD` : `t${b}USD`),

    url: (p, n) =>
      `${base('/proxy/bitfinex', 'https://api-pub.bitfinex.com')}/v2/candles/trade:1m:${p}/hist?limit=${n}`,

    // Bitfinex: [mts(ms), open, close, high, low, volume]
    parse: (j) => {
      if (Array.isArray(j) && j[0] === 'error') {
        throw new Error('unexpected response');
      }

      return arr(j).map((r) => c(r[0], r[1], r[3], r[4], r[2], r[5]));
    },
  },
  {
    id: 'gemini',
    name: 'Gemini',
    quote: 'USD',
    pair: (b) => b.toLowerCase() + 'usd',
    pollMs: 4000,
    url: (p) => `https://api.gemini.com/v2/candles/${p}/1m`,
    parse: (j) => arr(j).map((r) => c(r[0], r[1], r[2], r[3], r[4], r[5])),
  },
  // {
  //   id: 'kucoin',
  //   name: 'KuCoin',
  //   quote: 'USDT',
  //   pair: (b) => b + '-USDT',
  //   url: (p) =>
  //     `https://api.kucoin.com/api/v1/market/candles?type=1min&symbol=${p}`,
  // // `${base('/proxy/kucoin', 'https://api.kucoin.com')}/api/v1/market/candles?type=1min&symbol=${p}`,
  //   // KuCoin row order is [time(s), open, CLOSE, high, low, volume, turnover]
  //   parse: (j) =>
  //     arr(j.data).map((r) => c(r[0] * 1000, r[1], r[3], r[4], r[2], r[5])),
  // },
  {
    id: 'kucoin',
    name: 'KuCoin',
    quote: 'USDT',
    pair: (b) => b + '-USDT',

    url: (p) =>
      `${base('/proxy/kucoin', 'https://api.kucoin.com')}/api/v1/market/candles?type=1min&symbol=${encodeURIComponent(p)}`,

    // KuCoin: [time(s), open, close, high, low, volume, turnover]
    parse: (j) =>
      arr(j.data).map((r) =>
        c(Number(r[0]) * 1000, r[1], r[3], r[4], r[2], r[5]),
      ),
  },
  {
    id: 'whitebit',
    name: 'WhiteBIT',
    quote: 'USDT',
    pair: (b) => b + '_USDT',
    url: (p, n) =>
      `${base('/proxy/whitebit', 'https://whitebit.com')}/api/v1/public/kline?market=${p}&interval=1m&limit=${n}`,
    // WhiteBIT row order is [time(s), open, CLOSE, high, low, volume, turnover]
    parse: (j) => {
      if (j.success === false) throw new Error('unexpected response');
      return arr(j.result).map((r) =>
        c(r[0] * 1000, r[1], r[3], r[4], r[2], r[5]),
      );
    },
  },
  {
    // UNTESTED against the live API: endpoint/params are from BTSE's docs, but the start/end units are not confirmed.
    id: 'btse',
    name: 'BTSE',
    quote: 'USDT',
    pair: (b) => b + '-USDT',
    url: (p, n) => {
      const end = Date.now();
      return `${base('/proxy/btse', 'https://api.btse.com')}/spot/api/v3.2/ohlcv?symbol=${p}&resolution=1&start=${end - n * 60000}&end=${end}`;
    },
    // [time, open, high, low, close, volume]
    parse: (j) => arr(j).map((r) => c(ts(r[0]), r[1], r[2], r[3], r[4], r[5])),
  },
  {
    // UNTESTED against the live API: the docs list the endpoint and params but not the response shape, so this
    // parser accepts either row arrays or objects and seconds/ms/ISO timestamps.
    id: 'coinjar',
    name: 'CoinJar',
    quote: 'AUD',
    pair: (b) => b + 'AUD',
    url: (p, n) => {
      const before = Math.floor(Date.now() / 1000) + 60;
      return `${base('/proxy/coinjar', 'https://data.exchange.coinjar.com')}/products/${p}/candles?interval=1m&after=${before - (n + 1) * 60}&before=${before}`;
    },
    parse: (j) =>
      arr(Array.isArray(j) ? j : j.candles || j.data || j.result).map((r) =>
        Array.isArray(r)
          ? c(ts(r[0]), r[1], r[2], r[3], r[4], r[5])
          : c(
              ts(r.time ?? r.timestamp ?? r.t),
              r.open ?? r.o,
              r.high ?? r.h,
              r.low ?? r.l,
              r.close ?? r.c,
              r.volume ?? r.v,
            ),
      ),
  },
];

// ---- CCXT mode ----------------------------------------------------------------------------------------------------
// Same shape as an EXCHANGES entry, so the existing 1-minute polling path in useSignalDesk works unchanged: it polls the
// backend's /api/candles (served from a live CCXT Pro feed) instead of each exchange's own REST endpoint. That also removes
// the need for every /proxy/* entry in vite.config.js — the server talks to the exchanges, so there is no CORS.
// Left side = this app's id, right side = CCXT's id. Anything not listed maps to itself.
export const CCXT_IDS = {
  binance2: 'binance', // "Binance Global" is the same exchange as binance; the second entry only existed to swap the proxy route
  coinbase: 'coinbaseexchange', // api.exchange.coinbase.com (the old Pro API) — CCXT's plain 'coinbase' is the Advanced Trade API
  gate: 'gate',
  cryptocom: 'cryptocom',
  htx: 'htx',
};

export function ccxtExchange(ex) {
  const ccxtId = CCXT_IDS[ex.id] || ex.id;
  return {
    id: `ccxt:${ex.id}`,
    baseId: ex.id,
    ccxt: true,
    ccxtId,
    name: `${ex.name} · CCXT`,
    quote: ex.quote,
    multi: false, // the server aggregates 1-minute candles only (no 1s klines through CCXT)
    pollMs: 2000, // cheap: the backend answers from memory, it is not an exchange request
    pair: (b) => `${b}/${ex.quote}`, // CCXT unified symbol
    url: (p, n) => apiUrl(`/api/candles?exchange=${encodeURIComponent(ccxtId)}&symbol=${encodeURIComponent(p)}&limit=${n}`),
    parse: (j) => arr(j && j.candles).map((k) => c(k.t, k.o, k.h, k.l, k.c, k.v)),
  };
}

// ---- CCXT consolidation --------------------------------------------------------------------------------------------
// In CCXT mode the server merges the SAME asset across every selected exchange into one candle series and runs the FastAPI
// indicators (ta Bollinger / StochRSI / ATR) + strategies on that. Prices in different currencies can't be merged, so only
// dollar-like quotes take part; the server picks BASE/USDT, BASE/USD or BASE/USDC per exchange, whichever it lists.
export const USD_LIKE = ['USD', 'USDT', 'USDC'];
export const DEFAULT_CCXT_EXCHANGES = ['coinbaseexchange', 'kraken', 'bitstamp', 'okx'];

// The hand-written EXCHANGES list, as CCXT ids, minus non-dollar venues (Upbit KRW, CoinJar AUD) and the duplicate Binance entry.
export function ccxtChoices() {
  const seen = new Map();
  EXCHANGES.forEach((e) => {
    if (!USD_LIKE.includes(e.quote)) return;
    const id = CCXT_IDS[e.id] || e.id;
    if (!seen.has(id)) seen.set(id, { id, name: e.name.replace(' Global', '').replace(' (Huobi)', '') });
  });
  return [...seen.values()];
}

// Synthetic "exchange" describing a consolidated selection, shaped like the entries above so Header / cards can read it.
export function aggExchange(ids) {
  const names = Object.fromEntries(ccxtChoices().map((c) => [c.id, c.name]));
  return {
    id: `agg:${ids.join('+')}`, baseId: 'agg', ccxt: true, agg: true, multi: false,
    name: ids.length === 1 ? (names[ids[0]] || ids[0]) : `${ids.length} exchanges`,
    quote: 'composite', pair: (b) => `${b} on ${ids.length} exchange${ids.length === 1 ? '' : 's'}`,
  };
}

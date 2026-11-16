import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { Resolver } from 'node:dns/promises';
import https from 'node:https';



// import dns from 'node:dns';

// MEXC, Crypto.com, WhiteBIT, BTSE and CoinJar's public REST APIs don't (or can't be relied on to) send CORS headers for direct browser requests, so the
// browser blocks the response before the app ever sees it — no client-side fix is possible for that. This proxies
// those five through the Vite dev server instead: the *server* fetches the exchange (no CORS involved) and hands
// the browser a same-origin response. Gate.io/Bitget/OKX/Coinbase/HTX/Upbit are fine as direct browser fetches and
// are left alone. Dev-only — a production build has no dev server, so a real deployment needs an equivalent
// serverless/reverse-proxy route for these five (see PROXIED_EXCHANGES in src/lib/exchanges.js).
const proxy = {
  // FastAPI backend (CCXT + ta + SQLite) — see ./backend. `ws: true` is what lets /ws/desk through.
  '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/ws': { target: 'ws://127.0.0.1:8000', ws: true, changeOrigin: true },
  '/proxy/mexc': {
    target: 'https://api.mexc.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/mexc/, ''),
  },
  '/proxy/cryptocom': {
    target: 'https://api.crypto.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/cryptocom/, ''),
  },
  '/proxy/whitebit': {
    target: 'https://whitebit.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/whitebit/, ''),
  },
  '/proxy/btse': {
    target: 'https://api.btse.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/btse/, ''),
  },
  '/proxy/coinjar': {
    target: 'https://data.exchange.coinjar.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/coinjar/, ''),
  },
  '/proxy/kucoin': {
    target: 'https://api.kucoin.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/kucoin/, ''),
  },
  '/proxy/bitfinex': {
    target: 'https://api-pub.bitfinex.com',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/bitfinex/, ''),
  },
  '/proxy/binance': {
    target: 'https://data-api.binance.vision',
    changeOrigin: true,
    rewrite: (p) => p.replace(/^\/proxy\/binance/, ''),
  },
  // '/proxy/binanceus': {
  //   target: 'https://api.binance.us',
  //   changeOrigin: true,
  //   rewrite: (p) => p.replace(/^\/proxy\/binanceus/, ''),
  // },
  //   '/proxy/binanceus': {
  //     target: 'https://binance.us',
  //     changeOrigin: true,
  //     secure: true,
  //     // Inject headers to bypass Cloudflare bot detection
  //     headers: {
  //       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
  //       'Accept': 'application/json',
  //       'Accept-Language': 'en-US,en;q=0.9',
  //     },
  //     rewrite: (p) => p.replace(/^\/proxy\/binanceus/, ''),
  //   },
};




function binanceUsMiddleware() {
  const resolver = new Resolver();

  // Use normal Cloudflare DNS instead of the filtered resolver
  // that was returning 0.0.0.0 for api.binance.us.
  resolver.setServers(['1.1.1.1']);

  return {
    name: 'binanceus-middleware',

    configureServer(server) {
      server.middlewares.use('/proxy/binanceus', async (req, res) => {
        try {
          if (req.method !== 'GET') {
            res.statusCode = 405;
            res.end('Method Not Allowed');
            return;
          }

          // Because this middleware is mounted at /proxy/binanceus,
          // req.url should now be /api/v3/...
          const path = req.url || '/';

          console.log('[binanceus] request:', path);

          // Resolve api.binance.us using 1.1.1.1
          const addresses = await resolver.resolve4('api.binance.us');

          if (!addresses.length) {
            throw new Error('No IPv4 address found for api.binance.us');
          }

          const ip = addresses[0];

          console.log('[binanceus] resolved:', ip);

          const upstream = https.request(
            {
              hostname: ip,
              port: 443,
              path,
              method: 'GET',

              // Critical:
              // TLS SNI must still say api.binance.us
              servername: 'api.binance.us',

              // CloudFront/Binance needs the correct Host header
              headers: {
                Host: 'api.binance.us',
                Accept: 'application/json',
              },
            },
            (upstreamRes) => {
              console.log('[binanceus] upstream:', upstreamRes.statusCode);

              // DO NOT forward Location.
              //
              // If Binance returns a redirect, don't expose that
              // redirect to the browser.
              if (
                upstreamRes.statusCode >= 300 &&
                upstreamRes.statusCode < 400
              ) {
                res.statusCode = 502;
                res.setHeader('Content-Type', 'application/json');
                res.end(
                  JSON.stringify({
                    error: 'Binance.US returned a redirect',
                    status: upstreamRes.statusCode,
                    location: upstreamRes.headers.location || null,
                  }),
                );

                upstreamRes.resume();
                return;
              }

              res.statusCode = upstreamRes.statusCode || 502;

              // Copy safe response headers, but NEVER Location.
              if (upstreamRes.headers['content-type']) {
                res.setHeader(
                  'Content-Type',
                  upstreamRes.headers['content-type'],
                );
              }

              if (upstreamRes.headers['content-length']) {
                res.setHeader(
                  'Content-Length',
                  upstreamRes.headers['content-length'],
                );
              }

              upstreamRes.pipe(res);
            },
          );

          upstream.on('error', (err) => {
            console.error('[binanceus] HTTPS error:', err);

            if (!res.headersSent) {
              res.statusCode = 502;
              res.setHeader('Content-Type', 'application/json');
              res.end(
                JSON.stringify({
                  error: 'Binance.US connection failed',
                  message: err.message,
                }),
              );
            }
          });

          upstream.end();
        } catch (err) {
          console.error('[binanceus] middleware error:', err);

          if (!res.headersSent) {
            res.statusCode = 502;
            res.setHeader('Content-Type', 'application/json');
            res.end(
              JSON.stringify({
                error: 'Binance.US middleware failed',
                message: err.message,
              }),
            );
          }
        }
      });
    },
  };
}


export default defineConfig({
  plugins: [react(), binanceUsMiddleware()],
  server: { proxy },
  preview: { proxy },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.mjs'],
    exclude: ['tests/loader.mjs', 'tests/register.mjs'],
  },
});



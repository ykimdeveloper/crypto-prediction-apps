# Signal Desk (React)

A React/Vite port of the original single-file `signal-desk-binance.html`. Same engine, same
strategies, same layout — split into modules and driven by a React hook instead of manual DOM
updates.

## Run it

```
npm install
npm run dev
```

Then open the printed local URL. **This must run locally** (or anywhere your own browser can
reach Binance directly) — it opens a live WebSocket from the browser to
`wss://stream.binance.com:9443` and calls Binance's public REST API to seed history. A hosted
sandbox with a locked-down network will not see live data.

## Structure

```
src/
  lib/
    constants.js   symbols, timeframes, engine parameters
    format.js       fmt / fmtRemaining / clampConf / symShort
    indicators.js   Bollinger, StochRSI, ATR — pure functions over closed candles
    strategy.js     lean/story classifier + the three per-window strategies
    engine.js       per symbol×timeframe candle state, kline ingestion, aggregation
  hooks/
    useSignalDesk.js  owns the WebSocket, REST seed, 500ms render tick, and log state
  components/
    Header.jsx      title, feed status dot, timeframe selector
    SymbolCard.jsx  spot/strike/book/indicators + the three strategy boxes
    StrategyBox.jsx one strategy box (Mid-Mkt Rev / Mom / End-Mkt)
    SignalLog.jsx   rolling log across all symbols and timeframes
    Footer.jsx      the "what this reads" explainer
  App.jsx / App.css / index.css / main.jsx
```

## What's the same as the original

- Five parallel candle engines per symbol (1s, 5s, 10s, 30s, 1m); 1s/1m come straight from
  Binance's `kline_1s`/`kline_1m` streams, 5s/10s/30s are aggregated client-side from closed 1s
  klines.
- Bollinger(20) / StochRSI(14) / ATR(15) on each engine, and the same virtual 15-candle
  contract with the mid-market → end-of-market switch at 6 candles remaining.
- The same three strategies (Mid-Market Mean Reversion, Mid-Market Momentum Continuation,
  End-of-Market) with identical thresholds and confidence heuristic.
- The confidence % is a heuristic, not a backtested win rate, and there's no real contract,
  order book, or P&L behind this — technical-analysis display only, not financial advice.

## What changed in the port

- DOM string-building (`innerHTML =`) became React components reading from a snapshot object.
- The module-level mutable `engines`/`book`/`globalLog` became `useRef`/`useState` inside
  `useSignalDesk`, with a 500ms interval driving re-renders the same way the original's
  `renderAll()` interval did.
- Signal-log pushes are returned from `evaluateAndLog` as data instead of mutating a global
  array, so the hook can fold them into React state.

## Exchanges and tickers

- Pick an exchange from the dropdown. **Binance** is the multi-timeframe view (1s/5s/10s/30s/1m, live WebSocket, bid/ask).
  Every other exchange is a **1-minute only** view that polls its public REST kline endpoint (adapters in `src/lib/exchanges.js`).
- Click ticker chips to add or remove them; each selected ticker gets its own full engine and card. Type any other base asset
  (e.g. `HYPE`) and press Enter. Selections persist in `localStorage`.
- Binance adds/removes tickers live with WebSocket SUBSCRIBE/UNSUBSCRIBE. Polled exchanges make one request per ticker every ~3s,
  so keep the list modest on rate-limited venues (Crypto.com ~10 req/s, Upbit ~10 req/s).
- Bid/ask is only shown for Binance. Upbit prices are in KRW and Coinbase in USD; pairs not listed on an exchange show "no data".

## Data source: CCXT (multi-exchange)
In **CCXT** mode the exchange dropdown becomes a multi-select ("N exchanges merged"). The server merges every ticked exchange's 1-minute
candles per ticker (volume-weighted by default) and runs the FastAPI indicators + strategies on the merged series. Each card shows one chip
per exchange (live / stale / n/a, price, offset from the merged price in bp). The box in the menu accepts any CCXT exchange id.
**FastAPI** mode is unchanged: one exchange, full server desk. **Custom routes** is unchanged: browser talks to each exchange directly.
Rebuild with `npm run build` if you serve `dist/`.

## v0.2 strategy rules (from the Bleeplorp 0.2 walkthrough)

Tunable constants live in `src/lib/constants.js` (`TIMING`, `MID`, `EOM_INDICATOR_WEIGHT`) and are mirrored 1:1 in
`backend/app/constants.py`. Change both together; `backend/tests/test_parity.py` checks the two sides agree.

- **Mid-market is momentum-only.** Mean reversion is still classified (the lean/story labels are unchanged) but `MID.revEnabled = false`
  means MID REV never buys. This is separate from the `midEnabled` UI toggle, which pauses both forks.
- **At least 6:00 left** (`TIMING.midEntryMinRemaining`).
- **Price already slightly on the lean's side of the strike, not exhausted:** signed distance from the strike, in sigmas of the remaining path
  (`ATR * sqrt(candles left)`), must be within `MID.sideMinZ`..`MID.sideMaxZ`. Momentum fork only.
- **The +20¢ must be likely:** `pHit = P(side wins) / (ask + tp + half-spread)` must be at least `MID.minHitProb`. Momentum fork only.
- **End-of-market indicators are a fading nudge:** weight `EOM_INDICATOR_WEIGHT` at 8:00, linearly to 0 at 2:00.

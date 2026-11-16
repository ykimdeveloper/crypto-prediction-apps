import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { TIMEFRAMES, AGG_TFS, CONTRACT_MS, STALE_MS, STRAT_LABEL } from '../lib/constants';
import { EXCHANGES, ccxtExchange, classifyFetchError } from '../lib/exchanges';
import {
  createEngine, setEngineCandles, aggregateFromSeconds, updateEngineFromKline,
  feedClosedIntoAggregate, evaluateAndLog, maybeRollContract, refreshStrike, livePrice,
} from '../lib/engine';
import { computeModuleSignals } from '../lib/strategy';
import { fmt, fmtRemaining } from '../lib/format';
import { createPaper, loadDone, saveDone, settle, apply as applyPaper, summarize } from '../lib/paper';

const BINANCE_WS = 'wss://stream.binance.com:9443/stream';
const toCandle = (r) => ({ t: r[0], o: +r[1], h: +r[2], l: +r[3], c: +r[4], v: +r[5] });
const WAIT = { signal: 'WAIT', conf: null, reason: 'Loading…' };
const emptyCard = (symbol, quote) => ({
  symbol, quote, spot: '—', strike: '—', dist: '—', book: null, spotBook: null, stoch: '—', bb: '—', atr: '—',
  module: '—', closesIn: '—', rev: WAIT, mom: WAIT, end: WAIT, failed: false,
});

// opts: { feeCents, midEnabled, exitCfg: { stopPct (0-1), tpEnd (cents) } } — all read live, so changing them never restarts a feed.
//       mode: 'custom' (default) = original per-exchange routes; 'ccxt' = same engine, but candles are polled from the FastAPI/CCXT backend;
//             'off' = ML · FastAPI is on: this hook stays idle (useServerDesk is the active one).
export function useSignalDesk(exchangeId, tickers, onRec, opts = {}) {
  const mode = opts.mode || 'custom';
  const enabled = mode === 'custom' || mode === 'ccxt'; // 'off' = the server (useServerDesk) is doing the work
  const ex = useMemo(() => {
    const base = EXCHANGES.find((e) => e.id === exchangeId);
    return mode === 'ccxt' ? ccxtExchange(base) : base;
  }, [exchangeId, mode]);
  const enginesRef = useRef({}); // { TICKER: { tfKey: engine } }
  const bookRef = useRef({});
  const failedRef = useRef(new Set());
  const paperRef = useRef(createPaper());
  const lastTickRef = useRef(0);
  const tickersRef = useRef(tickers);
  const syncRef = useRef(() => {});
  tickersRef.current = tickers;
  const onRecRef = useRef(onRec);
  onRecRef.current = onRec;
  const tfRef = useRef('1m'); // timeframe on screen; alerts only fire for this one
  const optsRef = useRef(opts);
  optsRef.current = opts;
  const syncPaperOpts = (p) => {
    const o = optsRef.current;
    p.feeCents = o.feeCents || 0;
    p.disabled = new Set(o.midEnabled === false ? ['rev', 'mom'] : []);
    p.exitCfg = o.exitCfg || null;
  };
  syncPaperOpts(paperRef.current);

  const [feed, setFeed] = useState('connecting');
  const [feedNote, setFeedNote] = useState('');
  const [activeTf, setActiveTf] = useState('1m'); // transcript: 20 one-minute bars, 15-minute contract
  const [log, setLog] = useState([]);
  const [tick, setTick] = useState(0);

  const pushLog = useCallback((sym, tfKey, entries) => {
    if (!entries.length) return;
    setLog((prev) => [
      ...entries.map((e) => ({
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        time: new Date(), sym, tf: tfKey, strat: STRAT_LABEL[e.stratKey], signal: e.signal, conf: e.conf,
      })),
      ...prev,
    ].slice(0, 80));
  }, []);

  // Everything transport-related lives here and is rebuilt when the exchange changes.
  useEffect(() => {
    enginesRef.current = {};
    bookRef.current = {};
    if (!enabled) return undefined; // ML · FastAPI on: no browser feeds, no timers
    failedRef.current = new Set();
    paperRef.current = createPaper(loadDone(ex.id));
    syncPaperOpts(paperRef.current);
    lastTickRef.current = 0;
    setLog([]);
    setFeed('connecting');
    setFeedNote('');

    let dead = false;
    let ws = null;
    let reconnectTimer = null;
    let pollTimer = null;
    let opened = false;
    const subs = new Set();
    const tfKeys = ex.multi ? TIMEFRAMES.map((t) => t.key) : ['1m'];
    const evalLog = (eng, tfMs, sym, tfKey) => {
      const { sig } = evaluateAndLog(eng, tfMs);
      const exits = [];
      const opened = applyPaper(paperRef.current, sym, tfKey, tfMs, eng, sig, exits);
      pushLog(sym, tfKey, opened);
      if (exits.length) pushLog(sym, tfKey, exits.map((x) => ({ stratKey: x.stratKey, signal: x.kind === 'stop' ? 'STOP LEVEL' : 'TAKE PROFIT', conf: null })));
      if (tfKey === tfRef.current && onRecRef.current) {
        opened.forEach((o) => onRecRef.current({ kind: 'buy', sym, tf: tfKey, ...o }));
        exits.forEach((x) => onRecRef.current({ ...x, tf: tfKey }));
      }
    };

    // ---------- Binance: REST seed + WebSocket (dynamic SUBSCRIBE / UNSUBSCRIBE) ----------
    async function seedBinance(base) {
      const pair = ex.pair(base);
      const get = (i, n) =>
        fetch(`https://api.binance.com/api/v3/klines?symbol=${pair}&interval=${i}&limit=${n}`).then((r) => {
          if (!r.ok) throw new Error(`${i} klines ${r.status}`);
          return r.json();
        });
      try {
        const [secRaw, minRaw] = await Promise.all([get('1s', 1000), get('1m', 300)]);
        const engs = enginesRef.current[base];
        if (dead || !engs) return;
        const sec = secRaw.map(toCandle);
        const min = minRaw.map(toCandle);
        const now = Date.now();
        if (sec.length && sec[sec.length - 1].t + 1000 > now) sec.pop();
        if (min.length && min[min.length - 1].t + 60000 > now) min.pop();
        setEngineCandles(engs['1s'], sec);
        AGG_TFS.forEach((tf) => setEngineCandles(engs[tf.key], aggregateFromSeconds(sec, tf.ms)));
        setEngineCandles(engs['1m'], min);
        Object.values(engs).forEach(refreshStrike); // the exact window-open candle may only exist now
        failedRef.current.delete(base);
      } catch (e) {
        console.warn('seed failed for', base, e);
        failedRef.current.add(base);
      }
    }

    function syncBinance() {
      if (!ws || ws.readyState !== 1) return;
      const want = new Set();
      tickersRef.current.forEach((b) => {
        const l = ex.pair(b).toLowerCase();
        want.add(l + '@kline_1s'); want.add(l + '@kline_1m'); want.add(l + '@bookTicker');
      });
      const add = [...want].filter((s) => !subs.has(s));
      const del = [...subs].filter((s) => !want.has(s));
      if (add.length) ws.send(JSON.stringify({ method: 'SUBSCRIBE', params: add, id: Date.now() }));
      if (del.length) ws.send(JSON.stringify({ method: 'UNSUBSCRIBE', params: del, id: Date.now() + 1 }));
      subs.clear();
      want.forEach((s) => subs.add(s));
    }

    function connectBinance() {
      ws = new WebSocket(BINANCE_WS);
      ws.onopen = () => {
        subs.clear();
        if (opened) Object.keys(enginesRef.current).forEach(seedBinance); // backfill after a reconnect gap
        opened = true;
        syncBinance();
      };
      ws.onmessage = (ev) => {
        if (dead) return;
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        const d = msg && msg.data;
        if (!d || !msg.stream || !d.s) return;
        const base = d.s.replace(/USDT$/, '');
        const engs = enginesRef.current[base];
        if (!engs) return;
        lastTickRef.current = Date.now();
        setFeed('live');
        if (msg.stream.endsWith('@kline_1s') && d.k) {
          updateEngineFromKline(engs['1s'], d.k);
          evalLog(engs['1s'], 1000, base, '1s');
          if (d.k.x) {
            const closed = { o: +d.k.o, h: +d.k.h, l: +d.k.l, c: +d.k.c, v: +d.k.v, t: d.k.t };
            AGG_TFS.forEach((tf) => {
              feedClosedIntoAggregate(engs[tf.key], closed, tf.ms);
              evalLog(engs[tf.key], tf.ms, base, tf.key);
            });
          }
        } else if (msg.stream.endsWith('@kline_1m') && d.k) {
          updateEngineFromKline(engs['1m'], d.k);
          evalLog(engs['1m'], 60000, base, '1m');
        } else if (msg.stream.endsWith('@bookTicker') && d.b && d.a && bookRef.current[base]) {
          bookRef.current[base].bid = parseFloat(d.b);
          bookRef.current[base].ask = parseFloat(d.a);
        }
      };
      ws.onclose = () => {
        if (dead) return;
        setFeed('stale');
        reconnectTimer = setTimeout(connectBinance, 3000);
      };
      ws.onerror = () => { if (!dead) setFeed('stale'); };
    }

    // ---------- other exchanges: poll the public 1m kline endpoint ----------
    async function poll(base) {
      const engs = enginesRef.current[base];
      if (!engs) return;
      try {
        const r = await fetch(ex.url(ex.pair(base), 100));
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const cs = ex.parse(await r.json()).sort((a, b) => a.t - b.t);
        if (dead || enginesRef.current[base] !== engs) return;
        const now = Date.now();
        const closed = cs.filter((k) => k.t + 60000 <= now);
        const forming = cs.find((k) => k.t + 60000 > now) || null;
        const eng = engs['1m'];
        setEngineCandles(eng, closed);
        eng.current = forming || closed[closed.length - 1] || null;
        eng.lastTickAt = now;
        maybeRollContract(eng, now);
        refreshStrike(eng);
        evalLog(eng, 60000, base, '1m');
        failedRef.current.delete(base);
        lastTickRef.current = now;
        setFeed('live');
        setFeedNote('');
      } catch (e) {
        console.warn(ex.name, base, e);
        failedRef.current.add(base);
        setFeedNote(`${ex.name}: ${base} failed — ${classifyFetchError(e, ex)}`);
      }
    }

    // ---------- reconcile engines with the selected tickers ----------
    function sync() {
      const want = tickersRef.current;
      const fresh = [];
      want.forEach((b) => {
        if (enginesRef.current[b]) return;
        enginesRef.current[b] = Object.fromEntries(tfKeys.map((k) => [k, createEngine(TIMEFRAMES.find((t) => t.key === k).ms)]));
        bookRef.current[b] = { bid: null, ask: null };
        fresh.push(b);
      });
      Object.keys(enginesRef.current).forEach((b) => {
        if (want.includes(b)) return;
        delete enginesRef.current[b];
        delete bookRef.current[b];
        failedRef.current.delete(b);
      });
      if (ex.multi) {
        fresh.forEach(seedBinance);
        syncBinance();
      } else {
        fresh.forEach(poll);
      }
    }
    syncRef.current = sync;

    if (ex.multi) connectBinance();
    else pollTimer = setInterval(() => tickersRef.current.forEach(poll), ex.pollMs || 3000);
    sync();

    const renderTimer = setInterval(() => {
      if (lastTickRef.current && Date.now() - lastTickRef.current > STALE_MS) setFeed('stale');
      settle(paperRef.current, enginesRef.current, Date.now());
      if (paperRef.current.dirty) saveDone(ex.id, paperRef.current);
      setTick((t) => t + 1);
    }, 500);

    return () => {
      dead = true;
      clearInterval(renderTimer);
      clearInterval(pollTimer);
      clearTimeout(reconnectTimer);
      if (ws) { try { ws.close(); } catch { /* noop */ } }
    };
  }, [ex, pushLog, enabled]);

  // ticker list changed: add/remove engines and subscriptions
  useEffect(() => { syncRef.current(); }, [tickers]);

  const tf = ex.multi ? activeTf : '1m';
  tfRef.current = tf;
  const tfMs = TIMEFRAMES.find((t) => t.key === tf).ms;

  const cards = useMemo(() => tickers.map((base) => {
    const eng = enginesRef.current[base] && enginesRef.current[base][tf];
    if (!eng) return emptyCard(base, ex.quote);
    const price = livePrice(eng);
    const strike = eng.contractStrike;
    const bk = bookRef.current[base] || {};
    const sig = eng.signals || computeModuleSignals(eng, tfMs);
    const failed = failedRef.current.has(base);
    const closesIn = eng.contractStart != null
      ? fmtRemaining(eng.contractStart + CONTRACT_MS - Date.now()) : '—';
    return {
      symbol: base,
      quote: ex.quote,
      failed,
      spot: fmt(price),
      strike: fmt(strike),
      dist: failed && price == null
        ? `no data — ${ex.pair(base)} may not be listed on ${ex.name}`
        : price != null && strike != null
        ? `${fmt(Math.abs(price - strike))} ${price - strike >= 0 ? 'above' : 'below'} strike` : '—',
      book: sig.book || null, // modeled Up/Down contract book in cents (there is no Kalshi contract)
      spotBook: ex.multi ? (bk.bid != null && bk.ask != null ? `${fmt(bk.bid)} / ${fmt(bk.ask)}` : '—') : null,
      stoch: eng.ind.stoch
        ? `${eng.ind.stoch.k.toFixed(0)} ${eng.ind.stoch.traj === 'Rising' ? '↑' : eng.ind.stoch.traj === 'Falling' ? '↓' : '→'}` : '—',
      bb: eng.ind.bb ? eng.ind.bb.state.replace('_', ' ') : '—',
      atr: eng.ind.atr
        ? `${fmt(eng.ind.atr.atr)}${eng.ind.atr.state === 'Quiet/Asleep' ? ' (asleep)' : ''}${sig.story ? ' · ' + (sig.story === 'Mean Reversion' ? 'revert' : sig.story === 'Momentum Continuation' ? 'mom' : 'mixed') + ' ' + (sig.lean > 0.05 ? '↑' : sig.lean < -0.05 ? '↓' : '→') : ''}`
        : '—',
      lean: sig.story ? (sig.lean >= 0 ? '+' : '') + sig.lean.toFixed(2) : '—',
      story: sig.story || '—',
      atrState: eng.ind.atr ? (eng.ind.atr.state === 'Quiet/Asleep' ? 'Asleep' : 'Active') : '—',
      module: sig.module,
      closesIn,
      rev: sig.rev, mom: sig.mom, end: sig.end,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [tick, tickers, ex, tf]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const paperStats = useMemo(() => summarize(paperRef.current, tf, tickers, Date.now()), [tick, tf, tickers]);

  const tfNote = `15-min contract · indicators on ${tf} candles`;

  return { paperStats, exchange: ex, feed, feedNote, activeTf, setActiveTf, tfNote, cards, log };
}

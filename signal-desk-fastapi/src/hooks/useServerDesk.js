import { useEffect, useMemo, useRef, useState } from 'react';
import { EXCHANGES, ccxtExchange, aggExchange } from '../lib/exchanges';
import { wsUrl } from '../lib/backend';
import { serverCard, emptyCard, EMPTY_PAPER } from '../lib/serverCards';

// FastAPI mode: the backend runs CCXT Pro, the ta indicators, the three strategies and paper trading, and writes SQLite.
// This hook only opens /ws/desk, tells the server which tickers + settings to use, and shapes what comes back into the same
// return value as useSignalDesk, so the rest of the app doesn't know which mode it is in.
// opts: { enabled, feeCents, midEnabled, exitCfg } — same fields as useSignalDesk; changing them never reconnects.
//       exchanges: string[] of CCXT ids — when given (CCXT mode), the server merges every one of them into a single candle
//       series per ticker before running the indicators. Changing the selection reconnects (debounced, so ticking several
//       boxes in a row opens one connection, not five).
const useDebounced = (value, ms) => {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
};

export function useServerDesk(exchangeId, tickers, onRec, opts = {}) {
  const multi = Array.isArray(opts.exchanges);
  const aggKey = useDebounced(multi ? [...opts.exchanges].sort().join(',') : '', 700);
  const noneSelected = multi && !aggKey;
  const enabled = !!opts.enabled && !noneSelected;
  const ex = useMemo(
    () => (multi && aggKey ? aggExchange(aggKey.split(',')) : ccxtExchange(EXCHANGES.find((e) => e.id === exchangeId))),
    [exchangeId, multi, aggKey],
  );
  const [msg, setMsg] = useState(null);
  const [feed, setFeed] = useState('connecting');
  const [feedNote, setFeedNote] = useState('');
  const [log, setLog] = useState([]);
  const [tick, setTick] = useState(0);
  const wsRef = useRef(null);
  const tickersRef = useRef(tickers);
  tickersRef.current = tickers;
  const onRecRef = useRef(onRec);
  onRecRef.current = onRec;
  const optsRef = useRef(opts);
  optsRef.current = opts;

  const send = (obj) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj));
  };
  const sendConfig = () => {
    const o = optsRef.current;
    send({ op: 'config', feeCents: o.feeCents, midEnabled: o.midEnabled, exitCfg: o.exitCfg || null });
  };
  const sendTickers = () => send({ op: 'subscribe', symbols: tickersRef.current });

  useEffect(() => {
    if (!enabled) return undefined;
    setMsg(null);
    setLog([]);
    setFeed('connecting');
    setFeedNote('');
    let dead = false;
    let timer = null;
    let everOpened = false;

    const connect = () => {
      const q = ex.agg ? `exchanges=${encodeURIComponent(aggKey)}` : `exchange=${encodeURIComponent(ex.ccxtId)}&quote=${encodeURIComponent(ex.quote)}`;
      const ws = new WebSocket(wsUrl(`/ws/desk?${q}`));
      wsRef.current = ws;
      ws.onopen = () => { everOpened = true; sendConfig(); sendTickers(); };
      ws.onmessage = (ev) => {
        if (dead) return;
        let m;
        try { m = JSON.parse(ev.data); } catch { return; }
        if (m.type === 'state') {
          setMsg(m);
          setFeed(m.feed);
          setFeedNote(m.note || '');
        } else if (m.type === 'log') {
          setLog((prev) => [
            ...m.entries.map((e) => ({ id: `${e.ts}-${Math.random().toString(36).slice(2)}`, time: new Date(e.ts), sym: e.sym, tf: e.tf, strat: e.strat, signal: e.signal, conf: e.conf })),
            ...prev,
          ].slice(0, 80));
        } else if (m.type === 'rec') {
          if (onRecRef.current) onRecRef.current(m.rec);
        } else if (m.type === 'error') {
          setFeed('stale');
          setFeedNote(`FastAPI: ${m.detail}`);
        }
      };
      ws.onclose = () => {
        if (dead) return;
        setFeed('stale');
        setFeedNote(everOpened
          ? 'FastAPI backend disconnected — reconnecting…'
          : 'cannot reach the FastAPI backend — is `uvicorn app.main:app --port 8000` running in ./backend?');
        timer = setTimeout(connect, 3000);
      };
      ws.onerror = () => { /* onclose follows and carries the message */ };
    };
    connect();
    const t = setInterval(() => setTick((n) => n + 1), 500); // keeps the "closes in" clock moving between server pushes

    return () => {
      dead = true;
      clearTimeout(timer);
      clearInterval(t);
      const ws = wsRef.current;
      wsRef.current = null;
      if (ws) { try { ws.close(); } catch { /* noop */ } }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ex, enabled, aggKey]);

  useEffect(() => { if (enabled) sendTickers(); }, [tickers, enabled]); // eslint-disable-line react-hooks/exhaustive-deps
  const { feeCents, midEnabled, exitCfg } = opts;
  useEffect(() => { if (enabled) sendConfig(); }, [enabled, feeCents, midEnabled, exitCfg && exitCfg.stopPct, exitCfg && exitCfg.tpEnd]); // eslint-disable-line react-hooks/exhaustive-deps

  const cards = useMemo(() => tickers.map((base) => {
    const snap = msg && msg.cards ? msg.cards.find((c) => c.symbol === base) : null;
    return snap ? serverCard(snap, ex.quote, ex.name, ex.pair(base)) : emptyCard(base, ex.quote);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }), [msg, tick, tickers, ex]);

  const merged = ex.agg ? ` · ${aggKey.split(',').length} exchange${aggKey.includes(',') ? 's' : ''} merged${msg && msg.aggMode ? ` (${msg.aggMode})` : ''}` : '';
  const tfNote = `15-min contract · indicators on 1m candles${merged} · computed by FastAPI${msg && msg.indicatorBackend ? ` (${msg.indicatorBackend})` : ''}`;
  return {
    paperStats: (msg && msg.paper) || EMPTY_PAPER, exchange: ex,
    feed: noneSelected ? 'stale' : feed,
    feedNote: noneSelected ? 'No exchange selected — open the Exchanges menu and tick at least one.' : feedNote,
    activeTf: '1m', setActiveTf: () => {}, tfNote, cards, log,
  };
}

import { useEffect, useState } from 'react';
import { useSignalDesk } from './hooks/useSignalDesk';
import { useServerDesk } from './hooks/useServerDesk';
import { MODES, resolveSource } from './lib/backend';
import { EXCHANGES, DEFAULT_CCXT_EXCHANGES } from './lib/exchanges';
import { TIMEFRAMES, DEFAULT_TICKERS, TICKER_PRESETS, PAPER } from './lib/constants';
import Header from './components/Header';
import TickerPicker from './components/TickerPicker';
import ExchangePicker from './components/ExchangePicker';
import SymbolCard from './components/SymbolCard';
import PaperPanel from './components/PaperPanel';
import SignalLog from './components/SignalLog';
import Footer from './components/Footer';
import AlertsPanel from './components/AlertsPanel';
import PolicyControl from './components/PolicyControl';
import { loadAlerts, saveAlerts, shouldAlert, shouldAlertExit, fireAlert, fireExitAlert } from './lib/alerts';
import { loadSettings, saveSettings } from './lib/settings';
import './App.css';

const load = (key, fallback) => {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
};

export default function App() {
  const [exchangeId, setExchangeId] = useState(() => {
    const saved = load('sd.exchange', 'binance');
    return EXCHANGES.some((e) => e.id === saved) ? saved : 'binance';
  });
  // Two independent controls: `mode` = which browser route feeds the JS engine (custom | ccxt); `ml` = FastAPI does the work instead.
  const [source, setSource] = useState(() => resolveSource(load('sd.mode', 'custom'), load('sd.ml', null)));
  const { mode, ml } = source;
  const setMode = (m) => setSource((cur) => ({ ...cur, mode: MODES.some((x) => x.id === m) ? m : 'custom' }));
  const setMl = (on) => setSource((cur) => ({ ...cur, ml: !!on }));
  const [tickers, setTickers] = useState(() => load('sd.tickers', DEFAULT_TICKERS));
  const [ccxtExchanges, setCcxtExchanges] = useState(() => {
    const saved = load('sd.ccxtExchanges', DEFAULT_CCXT_EXCHANGES);
    return Array.isArray(saved) ? saved.filter((x) => typeof x === 'string') : DEFAULT_CCXT_EXCHANGES;
  });
  const [alerts, setAlerts] = useState(loadAlerts);
  useEffect(() => { saveAlerts(alerts); }, [alerts]);
  const [settings, setSettings] = useState(loadSettings);
  useEffect(() => { saveSettings(settings); }, [settings]);
  const onRec = (rec) => {
    if (rec.kind === 'stop' || rec.kind === 'tp') { if (shouldAlertExit(alerts, rec)) fireExitAlert(alerts, rec); return; }
    if (shouldAlert(alerts, rec)) fireAlert(alerts, rec);
  };
  const deskOpts = {
    feeCents: settings.feeCents,
    midEnabled: settings.midEnabled,
    exitCfg: { stopPct: alerts.exits.stopPct / 100, tpEnd: alerts.exits.tpEnd },
  };
  // Both hooks are always called (rules of hooks); only one of them opens connections.
  //   ML off -> useSignalDesk: browser engine; `mode` picks the route (custom = each exchange directly, ccxt = candles from FastAPI/CCXT)
  //   ML on  -> useServerDesk: FastAPI + CCXT Pro on ANY CCXT exchange(s) (one or several, merged per ticker), ta indicators,
  //             strategies, paper trading, SQLite and the ML ↑ probability all on the server
  const local = useSignalDesk(exchangeId, tickers, onRec, { ...deskOpts, mode: ml ? 'off' : mode });
  const server = useServerDesk(exchangeId, tickers, onRec, { ...deskOpts, enabled: ml, exchanges: ccxtExchanges });
  const { paperStats, exchange, feed, feedNote, activeTf, setActiveTf, tfNote, cards, log } = ml ? server : local;

  useEffect(() => {
    try {
      localStorage.setItem('sd.exchange', JSON.stringify(exchangeId));
      localStorage.setItem('sd.tickers', JSON.stringify(tickers));
      localStorage.setItem('sd.mode', JSON.stringify(mode));
      localStorage.setItem('sd.ml', JSON.stringify(ml));
      localStorage.setItem('sd.ccxtExchanges', JSON.stringify(ccxtExchanges));
    } catch { /* storage unavailable */ }
  }, [exchangeId, tickers, mode, ml, ccxtExchanges]);

  return (
    <div className="wrap">
      <Header
        exchanges={EXCHANGES} exchange={exchange} setExchangeId={setExchangeId} mode={mode} setMode={setMode} modes={MODES}
        ml={ml} setMl={setMl} policyControl={<PolicyControl enabled={ml} />}
        exchangePicker={ml ? <ExchangePicker selected={ccxtExchanges} setSelected={setCcxtExchanges} /> : null}
        feed={feed} feedNote={feedNote} activeTf={activeTf} setActiveTf={setActiveTf}
        tfNote={tfNote} timeframes={TIMEFRAMES}
      />
      <TickerPicker tickers={tickers} setTickers={setTickers} presets={TICKER_PRESETS} />

      <PaperPanel stats={paperStats} minConf={PAPER.minConf} tf={exchange.multi ? activeTf : '1m'} settings={settings} setSettings={setSettings} />

      <AlertsPanel alerts={alerts} setAlerts={setAlerts} tickers={tickers} />

      <div className="sym-grid">
        {cards.map((card) => (
          <SymbolCard key={card.symbol} card={card} showMid={settings.midEnabled} onRemove={() => setTickers((cur) => cur.filter((t) => t !== card.symbol))} />
        ))}
      </div>
      {cards.length === 0 && <p className="empty">No tickers selected. Click a ticker above to start its feed.</p>}

      <SignalLog log={log} />
      <Footer />
    </div>
  );
}

const TF_LABEL = { '1s': '1 second', '5s': '5 second', '10s': '10 second', '30s': '30 second', '1m': '1 minute' };

export default function Header({ exchanges, exchange, setExchangeId, exchangePicker, mode, setMode, modes, ml, setMl, policyControl, feed, feedNote, activeTf, setActiveTf, tfNote, timeframes }) {
  const feedText = feed === 'live' ? 'live' : feed === 'stale' ? 'feed stale — waiting' : 'connecting…';

  return (
    <header>
      <div className="title-block">
        <h1>Signal Desk</h1>
        <p>bollinger · stochastic rsi · atr — {exchange.multi ? 'live multi-timeframe engine' : '1-minute engine'} on {exchange.name.toLowerCase()}{ml ? ' · ML · FastAPI' : mode === 'ccxt' ? ' · via ccxt' : ''}</p>
      </div>
      <div className="controls">
        <span className="feed-status">
          <span className={`feed-dot ${feed === 'live' ? 'live' : feed === 'stale' ? 'stale' : ''}`} />
          <span>{feedText}</span>
        </span>
        {/* Browser-side route. Greyed out (not hidden) while ML · FastAPI is on: the server always fetches through CCXT. */}
        <div className={`seg${ml ? ' disabled' : ''}`} role="group" aria-label="Browser data route" aria-disabled={ml}
          title={ml ? 'Off while ML · FastAPI is on — the server fetches every exchange through CCXT' : undefined}>
          <span className="chip-label">Data</span>
          {modes.map((m) => (
            <button key={m.id} type="button" disabled={ml} className={`seg-btn${!ml && mode === m.id ? ' on' : ''}`} aria-pressed={!ml && mode === m.id} title={ml ? undefined : m.hint} onClick={() => setMode(m.id)}>
              {m.label}
            </button>
          ))}
        </div>
        <div className={`ml-toggle${ml ? ' on' : ''}`} title="Run the engine on the FastAPI server: any CCXT exchange, ta indicators, paper trading, SQLite and the ML ↑ probability">
          <button type="button" role="switch" aria-checked={ml} aria-label="ML on FastAPI" className={`switch${ml ? ' on' : ''}`} onClick={() => setMl(!ml)}>
            <span className="knob" />
          </button>
          <span className="ml-label" onClick={() => setMl(!ml)}>ML · FastAPI</span>
        </div>
        {policyControl}
        {exchangePicker || (
          <select value={exchange.baseId || exchange.id} onChange={(e) => setExchangeId(e.target.value)} aria-label="Exchange">
            {exchanges.map((e) => (
              <option key={e.id} value={e.id}>{e.name} — {e.multi && mode === 'custom' ? '1s to 1m' : '1m only'}</option>
            ))}
          </select>
        )}
        {exchange.multi && (
          <select value={activeTf} onChange={(e) => setActiveTf(e.target.value)} aria-label="Timeframe">
            {timeframes.map((tf) => <option key={tf.key} value={tf.key}>{TF_LABEL[tf.key]}</option>)}
          </select>
        )}
        <span className="tf-note">{tfNote}</span>
      </div>
      {feedNote && <div className="feed-note">{feedNote}</div>}
    </header>
  );
}

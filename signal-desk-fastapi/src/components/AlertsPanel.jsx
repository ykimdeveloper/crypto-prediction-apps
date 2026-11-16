import { fireAlert, fireExitAlert, unlockAudio } from '../lib/alerts';

const STRATS = [['rev', 'Mid-Mkt Rev'], ['mom', 'Mid-Mkt Mom'], ['end', 'End-Mkt']];
const MODES = [['bell', 'Bell only'], ['voice', 'Voice ("BTC buy up")'], ['both', 'Bell + voice']];

export default function AlertsPanel({ alerts, setAlerts, tickers }) {
  const patch = (p) => setAlerts((a) => ({ ...a, ...p }));
  const toggleTicker = (t) =>
    patch({ muted: alerts.muted.includes(t) ? alerts.muted.filter((x) => x !== t) : [...alerts.muted, t] });
  const toggleStrat = (k) => patch({ strats: { ...alerts.strats, [k]: !alerts.strats[k] } });
  const toggleEnabled = () => {
    if (!alerts.enabled) unlockAudio(); // browsers only allow sound after a click on the page
    patch({ enabled: !alerts.enabled });
  };
  const test = (signal) => fireAlert(alerts, { sym: tickers[0] || 'BTC', signal });
  const testExit = (kind) => fireExitAlert(alerts, { sym: tickers[0] || 'BTC', kind });
  const patchExits = (p) => patch({ exits: { ...alerts.exits, ...p } });
  const num = (v, lo, hi, dflt) => { const n = parseFloat(v); return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : dflt; };

  return (
    <div className="panel alerts">
      <h2>Alerts — new BUY recs and exit levels on the timeframe you're viewing</h2>
      <div className="alerts-row">
        <button type="button" role="switch" aria-checked={alerts.enabled} aria-label="Alerts on or off"
          className={`switch${alerts.enabled ? ' on' : ''}`} onClick={toggleEnabled}>
          <span className="knob" />
        </button>
        <span className="alerts-state">{alerts.enabled ? 'On' : 'Off'}</span>
        <select value={alerts.mode} onChange={(e) => patch({ mode: e.target.value })} aria-label="Alert sound">
          {MODES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
        </select>
        <button type="button" className="chip" onClick={() => test('BUY UP')}>Test up</button>
        <button type="button" className="chip" onClick={() => test('BUY DOWN')}>Test down</button>
      </div>
      <div className="alerts-row">
        <button type="button" className={`chip${alerts.exits.enabled ? ' on' : ''}`} aria-pressed={alerts.exits.enabled}
          onClick={() => patchExits({ enabled: !alerts.exits.enabled })}>Exit alerts</button>
        <label className="ctl" title="Alerts once per position when its bid falls to this % of your entry price. Stops are manual on Kalshi: press SELL and it fills at the best bid, not at this level.">
          <span className="chip-label">Stop at</span>
          <input type="number" className="num-in" min="1" max="99" step="1" value={alerts.exits.stopPct}
            onChange={(e) => patchExits({ stopPct: num(e.target.value, 1, 99, 75) })} />
          <span className="chip-label">% of entry</span>
        </label>
        <label className="ctl" title="End-of-market has no preset target, so this is your take-profit price. Mid-market forks alert at their own +15/+20¢ target.">
          <span className="chip-label">End-mkt TP</span>
          <input type="number" className="num-in" min="50" max="99.9" step="0.1" value={alerts.exits.tpEnd}
            onChange={(e) => patchExits({ tpEnd: num(e.target.value, 50, 99.9, 99) })} />
          <span className="chip-label">¢</span>
        </label>
        <button type="button" className="chip" onClick={() => testExit('stop')}>Test stop</button>
        <button type="button" className="chip" onClick={() => testExit('tp')}>Test take-profit</button>
      </div>
      <div className="chips">
        <span className="chip-label">Strategies</span>
        {STRATS.map(([k, l]) => (
          <button key={k} type="button" className={`chip${alerts.strats[k] ? ' on' : ''}`} aria-pressed={alerts.strats[k]} onClick={() => toggleStrat(k)}>{l}</button>
        ))}
      </div>
      <div className="chips">
        <span className="chip-label">Cryptos</span>
        {tickers.length === 0 && <span className="chip-label">add tickers above</span>}
        {tickers.map((t) => {
          const on = !alerts.muted.includes(t);
          return <button key={t} type="button" className={`chip${on ? ' on' : ''}`} aria-pressed={on} onClick={() => toggleTicker(t)}>{t}</button>;
        })}
        {tickers.length > 0 && (
          <>
            <button type="button" className="chip" onClick={() => patch({ muted: [] })}>All</button>
            <button type="button" className="chip" onClick={() => patch({ muted: [...tickers] })}>None</button>
          </>
        )}
      </div>
    </div>
  );
}

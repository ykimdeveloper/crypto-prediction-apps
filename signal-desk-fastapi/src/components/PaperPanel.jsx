import { fmtCents, fmtUsd } from '../lib/format';
import { PRICE } from '../lib/constants';

const NAMES = { rev: 'Mid-market Mean Reversion', mom: 'Mid-market Momentum Continuation', end: 'End-of-market' };
// wins/total, win %, average cents per contract — the same cell the reference panel uses
const cell = (t) => (t.n ? `${t.w}/${t.n} ${Math.round(t.pct * 100)}% ${fmtCents(t.avg)}` : '—');
const pnlCls = (v) => (v == null ? '' : v >= 0 ? 'up' : 'down');
const gateWin = (w) => `${w.label} ${w.n ? fmtCents(w.avg) : 'n/a'} n=${w.n}`;

export default function PaperPanel({ stats, minConf, tf, settings, setSettings }) {
  const showMid = !settings || settings.midEnabled;
  const fee = settings ? settings.feeCents : 0;
  return (
    <div className="panel paper">
      <h2>
        Paper · {tf} · +{PRICE.tpMmr}/+{PRICE.tpMmc}¢ · ≥{minConf}% · ≥{PRICE.minEntry}¢ · no stop · one rec while lit · fee {fee}¢
      </h2>
      {settings && (
        <div className="paper-controls">
          <label className="ctl" title="Flat taker fee per contract, charged on entry (and again on a stop exit). 0 = limit orders that rest and fill.">
            <span className="chip-label">Fee ¢/contract</span>
            <input type="number" className="num-in" min="0" max="10" step="0.1" value={settings.feeCents}
              onChange={(e) => { const v = parseFloat(e.target.value); setSettings((s) => ({ ...s, feeCents: Number.isFinite(v) ? Math.min(10, Math.max(0, v)) : 0 })); }} />
          </label>
          <button type="button" className={`chip${settings.midEnabled ? ' on' : ''}`} aria-pressed={settings.midEnabled}
            onClick={() => setSettings((s) => ({ ...s, midEnabled: !s.midEnabled }))}>
            Mid-market strategies {settings.midEnabled ? 'on' : 'off'}
          </button>
          <span className="chip-label">P&amp;L below is net of fees; stop replays fill at the bid, not the stop price</span>
        </div>
      )}
      <div className="paper-scroll">
        {['rev', 'mom', 'end'].filter((k) => showMid || k === 'end').map((k) => {
          const s = stats[k];
          return (
            <section key={k}>
              <div className="paper-title">
                {NAMES[k]}{' '}
                <span className={`gate ${s.gate.held ? 'bad' : 'ok'}`}>
                  Gate {s.gate.held ? 'HELD' : 'open'} · {gateWin(s.gate.hold)} · {gateWin(s.gate.rel)}
                  {s.gate.held ? ' — shadow-trading until it releases' : ''}
                </span>
              </div>
              <div className="paper-grid head">
                <span /><span>ALL</span><span>24H</span><span>2H</span><span>24H PNL</span><span>ALL PNL</span><span>REC PNL</span><span>SL</span><span>TP</span><span>STRK</span>
              </div>
              {s.rows.map((r) => (
                <div className="paper-grid" key={r.label}>
                  <span className="lab">{r.label}</span>
                  <span>{cell(r.all)}</span><span>{cell(r.d24)}</span><span>{cell(r.h2)}</span>
                  <span className={pnlCls(r.d24.n ? r.d24.sum : null)}>{r.d24.n ? fmtUsd(r.d24.sum) : '—'}</span>
                  <span className={pnlCls(r.all.n ? r.all.sum : null)}>{r.all.n ? fmtUsd(r.all.sum) : '—'}</span>
                  <span className={pnlCls(r.rec)}>{r.rec != null ? fmtUsd(r.rec) : '—'}</span>
                  <span>{r.sl}</span><span>{r.tp}</span>
                  <span className={r.all.streak.endsWith('W') ? 'up' : r.all.streak.endsWith('L') ? 'down' : ''}>{r.all.streak}</span>
                </div>
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}

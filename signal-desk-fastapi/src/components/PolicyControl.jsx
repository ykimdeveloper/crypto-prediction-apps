import { useEffect, useState } from 'react';
import { getPolicy, setPolicy, policyWarning } from '../lib/policy';

const OPTIONS = [
  { id: 'original', label: 'Original', hint: 'Your algorithm exactly as before' },
  { id: 'gated', label: 'Gated', hint: 'Keep a BUY only if the trained 15-minute model backs its direction; otherwise WAIT' },
];

// Needs the server (the model lives there), so it is greyed out — not hidden — while ML · FastAPI is off.
export default function PolicyControl({ enabled }) {
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    if (!enabled) return undefined;
    let dead = false;
    const pull = async () => {
      try { const j = await getPolicy(); if (!dead) { setSt(j); setErr(''); } } catch (e) { if (!dead) setErr(e.message); }
    };
    pull();
    const t = setInterval(pull, 15000);
    return () => { dead = true; clearInterval(t); };
  }, [enabled]);

  const choose = async (id) => {
    try { setSt(await setPolicy(id)); setErr(''); } catch (e) { setErr(e.message); }
  };

  const off = !enabled || !!err;
  const current = st ? st.policy : 'original';
  const warn = err || policyWarning(st);
  return (
    <>
      <div className={`seg${off ? ' disabled' : ''}`} role="group" aria-label="Signal policy" aria-disabled={off}
        title={!enabled ? 'Needs ML · FastAPI on — the 15-minute model runs on the server' : undefined}>
        <span className="chip-label">Signals</span>
        {OPTIONS.map((o) => (
          <button key={o.id} type="button" disabled={off} className={`seg-btn${!off && current === o.id ? ' on' : ''}`}
            aria-pressed={!off && current === o.id} title={off ? undefined : o.hint} onClick={() => choose(o.id)}>
            {o.label}
          </button>
        ))}
      </div>
      {enabled && warn && <span className="policy-note" title={warn}>{warn}</span>}
    </>
  );
}

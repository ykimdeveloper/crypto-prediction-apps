import { useEffect, useRef, useState } from 'react';
import { ccxtChoices, USD_LIKE } from '../lib/exchanges';
import { apiUrl } from '../lib/backend';

const CHOICES = ccxtChoices();
const MAX_EXCHANGES = 25; // backend MAX_AGG_EXCHANGES default; it rejects a bigger set

// Exchange multi-select for ML · FastAPI mode. The top of the list is the venues this app already knows by name; once the
// backend answers /api/exchanges, EVERY exchange CCXT supports is listed below them (searchable). Tick one for a single
// exchange, several to merge them into one candle series per ticker.
export default function ExchangePicker({ selected, setSelected }) {
  const [open, setOpen] = useState(false);
  const [all, setAll] = useState({ ids: [], pro: [], loaded: false });
  const [query, setQuery] = useState('');
  const box = useRef(null);

  useEffect(() => {
    if (!open || all.loaded) return;
    fetch(apiUrl('/api/exchanges')).then((r) => r.json())
      .then((j) => setAll({ ids: j.exchanges || [], pro: j.pro || [], loaded: true }))
      .catch(() => { /* backend down: the known venues still work */ });
  }, [open, all.loaded]);
  useEffect(() => {
    if (!open) return undefined;
    const away = (e) => { if (box.current && !box.current.contains(e.target)) setOpen(false); };
    const esc = (e) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', away);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', away); document.removeEventListener('keydown', esc); };
  }, [open]);

  const known = new Set(CHOICES.map((c) => c.id));
  const proSet = new Set(all.pro);
  const rest = all.ids.filter((id) => !known.has(id) && !selected.includes(id)).map((id) => ({ id, name: id }));
  const extras = selected.filter((id) => !known.has(id)).map((id) => ({ id, name: id }));
  const rows = [...CHOICES, ...extras, ...rest];
  const q = query.trim().toLowerCase();
  const shown = q ? rows.filter((r) => r.id.includes(q) || r.name.toLowerCase().includes(q)) : rows;
  const atMax = selected.length >= MAX_EXCHANGES;
  const toggle = (id) => setSelected((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : cur.length >= MAX_EXCHANGES ? cur : [...cur, id]));
  const addTyped = () => {
    const id = q.replace(/[^a-z0-9_]/g, '');
    if (id && !selected.includes(id) && (!all.loaded || all.ids.includes(id))) toggle(id);
    setQuery('');
  };

  return (
    <div className="xpick" ref={box}>
      <button type="button" className="xpick-btn" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        {selected.length === 0 ? 'Pick exchanges' : selected.length === 1 ? `${(CHOICES.find((c) => c.id === selected[0]) || { name: selected[0] }).name} ▾` : `${selected.length} exchanges merged ▾`}
      </button>
      {open && (
        <div className="xpick-pop" role="group" aria-label="Exchanges for the server engine">
          <div className="xpick-actions">
            <button type="button" className="chip" onClick={() => setSelected(CHOICES.map((c) => c.id).slice(0, MAX_EXCHANGES))}>Select popular</button>
            <button type="button" className="chip" onClick={() => setSelected([])}>Clear</button>
            <span className="xpick-count">{selected.length}/{MAX_EXCHANGES}</span>
          </div>
          <div className="picker-add">
            <input value={query} placeholder={all.loaded ? `search ${all.ids.length} CCXT exchanges` : 'search or type a CCXT id'} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addTyped()} />
          </div>
          <div className="xpick-list">
            {shown.map((r) => (
              <label key={r.id} className="xpick-row">
                <input type="checkbox" checked={selected.includes(r.id)} disabled={atMax && !selected.includes(r.id)} onChange={() => toggle(r.id)} />
                <span>{r.name}</span>
                <em>{r.name !== r.id ? r.id : ''}{proSet.has(r.id) ? ' · ws' : ''}</em>
              </label>
            ))}
            {shown.length === 0 && <p className="xpick-note">No CCXT exchange matches “{query}”.</p>}
          </div>
          <p className="xpick-note">
            Tick one exchange to run the engine on it alone, or several to merge them: each ticker&apos;s candles are combined across the ticked exchanges, then Bollinger / StochRSI / ATR and the ML ↑ probability run on that series.
            {' '}{USD_LIKE.join(' / ')} pairs count as one dollar; an exchange that only lists the coin in KRW, EUR… is skipped and shows as n/a on the card. “ws” = live WebSocket feed, otherwise REST polling.
          </p>
        </div>
      )}
    </div>
  );
}

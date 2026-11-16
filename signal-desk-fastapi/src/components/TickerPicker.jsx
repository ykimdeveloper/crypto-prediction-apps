import { useState } from 'react';

export default function TickerPicker({ tickers, setTickers, presets }) {
  const [custom, setCustom] = useState('');
  const all = [...presets, ...tickers.filter((t) => !presets.includes(t))];
  const toggle = (t) => setTickers((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  const addCustom = () => {
    const t = custom.toUpperCase().replace(/[^A-Z0-9]/g, '');
    if (t.length >= 2 && t.length <= 12 && !tickers.includes(t)) setTickers((cur) => [...cur, t]);
    setCustom('');
  };

  return (
    <div className="panel picker">
      <h2>Tickers — click to add or remove ({tickers.length} selected)</h2>
      <div className="chips">
        {all.map((t) => (
          <button key={t} type="button" className={`chip${tickers.includes(t) ? ' on' : ''}`} aria-pressed={tickers.includes(t)} onClick={() => toggle(t)}>
            {t}
          </button>
        ))}
      </div>
      <div className="picker-add">
        <input
          value={custom}
          placeholder="other ticker, e.g. HYPE"
          onChange={(e) => setCustom(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && addCustom()}
        />
        <button type="button" className="chip" onClick={addCustom}>Add</button>
        {tickers.length > 0 && <button type="button" className="chip" onClick={() => setTickers([])}>Clear all</button>}
      </div>
    </div>
  );
}

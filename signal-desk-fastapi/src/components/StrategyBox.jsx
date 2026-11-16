export default function StrategyBox({ title, sig }) {
  const dir = sig.signal.endsWith('UP') ? 'up' : sig.signal.endsWith('DOWN') ? 'down' : 'wait';

  return (
    <div className={`strat-box${dir === 'wait' ? '' : ' active-' + dir}`}>
      <div className="stitle">{title}</div>
      <div className={`sval ${dir}`}>{sig.signal}</div>
      <div className="sconf">{sig.conf ? `${sig.conf}%` : '—'}</div>
      <div className="sreason">{sig.reason || ''}</div>
    </div>
  );
}

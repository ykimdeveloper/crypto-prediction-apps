export default function SignalLog({ log }) {
  return (
    <div className="panel">
      <h2>Signal Log — all symbols · all timeframes</h2>
      <div className="log-list">
        {log.map((row) => (
          <div className="log-row" key={row.id}>
            <span>{row.time.toLocaleTimeString()}</span>
            <span>{row.sym}</span>
            <span>{row.tf}</span>
            <span>{row.strat}</span>
            <span className={`s ${row.signal === 'BUY UP' || row.signal === 'TAKE PROFIT' ? 'up' : 'down'}`}>
              {row.signal}
              {row.conf ? ` · ${row.conf}%` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

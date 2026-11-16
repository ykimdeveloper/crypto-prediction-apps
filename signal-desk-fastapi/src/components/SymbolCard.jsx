import StrategyBox from './StrategyBox';

export default function SymbolCard({ card, onRemove, showMid = true }) {
  return (
    <div className="sym-card">
      <div className="sym-head">
        <div className="sym-name">
          {card.symbol} <span className="quote">/ {card.quote}</span>
        </div>
        <div className="sym-meta">
          {card.module} · closes in {card.closesIn}
          <button type="button" className="remove" aria-label={`Remove ${card.symbol}`} onClick={onRemove}>✕</button>
        </div>
      </div>
      <div className="row">
        <div><span className="lbl">Spot</span><span className="val">{card.spot}</span></div>
        <div><span className="lbl">Strike</span><span className="val">{card.strike}</span></div>
      </div>
      <div className="dist">{card.dist}</div>
      {card.book && (
        <div className="book" title="Modeled contract book in cents — there is no Kalshi contract here">
          <span className="lbl">Book ¢</span>
          <span>Up <b>{card.book.up.bid.toFixed(1)}</b> / <b>{card.book.up.ask.toFixed(1)}</b></span>
          <span>Down <b>{card.book.down.bid.toFixed(1)}</b> / <b>{card.book.down.ask.toFixed(1)}</b></span>
        </div>
      )}
      {card.venues && (
        <div className="venues" title="Each exchange's latest price and its offset from the consolidated price the indicators run on">
          <span className="lbl">Exchanges {card.venues.filter((v) => v.status === 'live').length}/{card.venues.length} live</span>
          {card.venues.map((v) => (
            <span key={v.exchange} className={`venue ${v.status}`} title={v.error || v.symbol || ''}>
              {v.exchange}{v.price ? <> <b>{v.price}</b> <i>{v.devBps}</i></> : <> <i>{v.status === 'error' ? 'n/a' : '…'}</i></>}
            </span>
          ))}
        </div>
      )}
      <div className="ind-row">
        {card.spotBook != null && <span>{card.venues ? 'Best bid/ask' : 'Spot bid/ask'} <b>{card.spotBook}</b></span>}
        <span>Stoch <b>{card.stoch}</b></span>
        <span>BB <b>{card.bb}</b></span>
        <span>ATR <b>{card.atr}</b></span>
      </div>
      <div className="ind-row">
        <span>Lean <b>{card.lean}</b></span>
        <span>Story <b>{card.story}</b></span>
        <span>ATR state <b>{card.atrState}</b></span>
        {card.policy && (
          <span title={card.policy.note || `15-minute model P(up) for this window; a BUY needs it to back the direction by ${card.policy.margin}`}>
            Model ↑ <b>{card.policy.pUp != null ? `${Math.round(card.policy.pUp * 100)}%` : '…'}</b>
            {card.policy.vetoed.length > 0 && <> · vetoed <b>{card.policy.vetoed.join(', ')}</b></>}
          </span>
        )}
        {card.ml != null && <span title="h2o model: probability the close 15 candles from now is above the current close. Information only — it never gates a strategy.">ML ↑ <b>{card.ml}</b></span>}
      </div>
      <div className="strat-row">
        {showMid && <StrategyBox title="Mid-Mkt Rev" sig={card.rev} />}
        {showMid && <StrategyBox title="Mid-Mkt Mom" sig={card.mom} />}
        <StrategyBox title="End-Mkt" sig={card.end} />
      </div>
    </div>
  );
}

export default function Footer() {
  return (
    <footer>
      <b>Contract:</b> every symbol runs a clock-aligned <b>15-minute</b> virtual contract (:00/:15/:30/:45), whatever candle size the
      indicators use. Strike = the price the window opened at. There is no Kalshi here, so the Up/Down book is <b>modeled</b>:
      P(finish above strike) = Φ(distance ÷ (ATR × √candles-left × 1.4)), quoted in cents with a 1¢ spread — the 1.4 factor is a
      least-squares fit to the four Up/Down quotes in the reference screenshot and lands within ~5¢ of them, not exactly on them. Everything priced in cents (entry bands, take-profit, P&amp;L, gates)
      runs off that model, not a real order book.
      <br />
      <br />
      <b>Three strategies, each on its own clock.</b> <b>Mid-Market Mean Reversion</b> (a very low meter turning up, or a very high one
      turning down, near the matching Bollinger edge) buys the stretched side at 35–50¢, target +15¢. <b>Mid-Market Momentum
      Continuation</b> (meter mid/high and still rising, or mid/low and still falling) buys a side already at 50–78¢, target +20¢. Both
      only enter with ≥5:00 left, skip a sleeping market, and stop watching the target at 1:00. <b>End-of-Market</b> runs 8:00→2:00 left,
      buys the side price is on only when its ask is inside 85–98¢ (above 98¢ it just says HOLD), and rides to settlement. All three need
      ≥80% confidence, use no stop, and fire once per window. Before 8:00 the End-of-Market box still shows the side price is on and its confidence, but it stays WAIT until the window opens.
      <br />
      <br />
      <b>Fees and exits.</b> Paper results are net of a flat per-contract taker fee (the Fee setting; it is charged on entry, and again on a stop exit, while a take-profit is a resting limit order and settlement is free — set it to 0 to model limit fills). The optimizer's stop replays fill at the bid you could actually hit when you press sell ("best price above 1¢"), not at the stop price. Kalshi's 15-minute markets have no preset stop, so <b>exit alerts</b> ring once per position when its bid falls to the stop level (default 75% of entry) or reaches the take-profit level — your cue to press SELL yourself. The <b>Mid-market strategies</b> switch pauses both forks (no recs, no paper trades, no alerts).
      <br />
      <br />
      <b>Gates</b> are strategy-wide and judged on average cents per contract, not win rate: mid-market forks hold when the trailing 1h
      average is ≤ −2¢ and release when the trailing 30m average is ≥ +2.5¢; End-of-Market holds at a 2h average ≤ −2¢ and releases when the
      1h average is positive. A held strategy keeps paper-trading in the background so its stats can recover — it just stops lighting up.
      The video says the mid-market release is "+25¢ … I think"; that is close to unreachable next to a +15/+20¢ target, so 2.5¢ is used
      (change <code>PAPER.gates.mid.releaseAtLeast</code> in <code>constants.js</code> to follow it literally). The <b>SL / TP / REC PNL</b>{' '}
      columns come from an optimizer that replays every completed result's recorded price path under a grid of stop-losses (% of entry)
      and take-profits and shows the best rule — it is a backtest of your own history, so it overfits on small samples (it needs 8+ results).
      <br />
      <br />
      <b>Data route and ML · FastAPI.</b> Two independent controls. <b>Custom routes</b> / <b>CCXT</b> only matter while ML is off: both keep all indicator and strategy logic in the browser; Custom routes is the original behaviour, CCXT polls 1-minute candles from the FastAPI backend, which holds a live CCXT Pro feed per exchange (no CORS, no per-exchange proxy, one <code>CCXT_IDS</code> table in <code>lib/exchanges.js</code>; 1-minute only, even for Binance). Turning on <b>ML · FastAPI</b> moves the logic to <code>backend/</code> too and greys those two buttons out, because the server always fetches through CCXT: pick any exchange CCXT supports (one, or several merged into one candle series), and Bollinger, Stochastic RSI and ATR run via the <code>ta</code> package, with the same three strategies, paper trading, SQLite storage (<code>candles</code>, <code>indicators</code>, <code>ticker_snapshots</code>, <code>signals</code>, <code>paper_trades</code>) and the ML ↑ probability once a model is trained. The browser just draws what the server streams. Server-mode paper history lives in SQLite, not in this browser, and is separate from the browser modes.
      <br />
      <br />
      <b>Exchanges:</b> Binance runs the full multi-timeframe engine over its WebSocket; every other exchange is 1-minute only, polled over
      public REST. <b>MEXC, Crypto.com, WhiteBIT, BTSE and CoinJar don't send (usable) CORS headers to browsers</b>, so <code>npm run dev</code> routes those five
      through Vite's dev proxy instead (see <code>vite.config.js</code> / <code>PROXIED_EXCHANGES</code> in <code>lib/exchanges.js</code>) —
      that only exists in dev, so a real deployment needs an equivalent proxy route for those five. BTSE and CoinJar are wired from their docs but haven't been tested against the live APIs. A failed ticker's note now says which
      of rate limit / unlisted pair / CORS it actually was; check the browser console for the full error too. Confidence % is a heuristic,
      not a backtested win rate. Paper history lives in this browser only (localStorage). Technical-analysis display only — not a trading
      system, and not financial advice.
    </footer>
  );
}

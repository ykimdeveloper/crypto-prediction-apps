export const DEFAULT_TICKERS = ['BTC', 'ETH', 'SOL', 'XRP'];
export const TICKER_PRESETS = [
  'BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'DOGE', 'ADA', 'AVAX', 'LINK', 'LTC',
  'DOT', 'TRX', 'SUI', 'TON', 'NEAR', 'ATOM', 'ARB', 'OP', 'APT', 'PEPE',
];

export const TIMEFRAMES = [
  { key: '1s', ms: 1000 },
  { key: '5s', ms: 5000 },
  { key: '10s', ms: 10000 },
  { key: '30s', ms: 30000 },
  { key: '1m', ms: 60000 },
];

// timeframes with no native Binance kline stream — built by aggregating closed 1s klines
export const AGG_TFS = TIMEFRAMES.filter((t) => ['5s', '10s', '30s'].includes(t.key));

// ---- Contract: ALWAYS 15 real minutes, clock-aligned (:00/:15/:30/:45), whatever candle size drives the indicators.
// (Old version used 15 *candles*, which was a 150-second "contract" on the 10s timeframe.)
export const MIN_MS = 60000;
export const CONTRACT_MS = 15 * MIN_MS;

// ---- Strategy time windows, in minutes REMAINING on the contract (from the transcript / reference screenshot).
export const TIMING = {
  midEntryMinRemaining: 5, // MMR + MMC: reference shows "Under five minutes" once < 5:00 is left
  midManageUntil: 1,       // header: "mid-market (to 1:00)" -> take-profit is only watched until 1:00 remains
  eomMaxRemaining: 8,      // header: "end-of-market (8:00 -> 2:00)"
  eomMinRemaining: 2,
};

// ---- Contract price rules, in cents on a 0-100 scale (Kalshi-style Up/Down contracts)
export const PRICE = {
  minEntry: 35,               // header "≥35¢": never buy a contract cheaper than this on a mid-market fork
  mmrBand: [35, 50],          // reversion buys the cheap, stretched side ("35-40 cent positions") expecting it to snap back
  mmcBand: [50, 78],          // continuation buys a side that's already solid, so +20¢ still fits under 98¢
  eomBand: [85, 98],          // end-of-market buys only inside 85-98¢; above 98¢ there is nothing left to earn
  tpMmr: 15,                  // header "+15/+20¢"
  tpMmc: 20,
};

// Modeled contract book. There is no Kalshi here, so Up/Down prices are modeled as the chance of finishing above/below
// the strike: Phi(distance / (ATR * sqrt(candles left) * sigmaFudge)). 1.4 is the least-squares fit to the four Up/Down
// quotes in your second screenshot. One constant can't match all four (the real book's implied vol varies by coin, roughly
// 1.2x to 1.85x here), so expect the modeled Up price to land within ~5¢ of a real quote, not on it.
export const BOOK = { sigmaFudge: 1.4, halfSpread: 0.5 };
// End-of-market confidence uses an even more conservative tail so it sits a few points under the book price (as in the screenshot).
export const EOM_CONF_SIGMA = 2.0;

export const BB_PERIOD = 20;
export const STOCH_PERIOD = 14;
export const RSI_PERIOD = 14;
export const ATR_PERIOD = 15;
// "asleep" floor: 7 bps of price per MINUTE. Scaled by sqrt(candle/1m) so sub-minute candles aren't judged against a 1m floor.
export const ATR_FLOOR_BPS_PER_MIN = 0.0007;

export const STALE_MS = 6000;
export const MAX_CANDLES = 400;

export const STRAT_LABEL = { rev: 'MID REV', mom: 'MID MOM', end: 'END-MKT' };
export const STRAT_KEYS = ['rev', 'mom', 'end'];

// ---- Paper trading + gates (transcript: gate on average cents-per-contract, strategy-wide, shadow trading continues while held)
export const PAPER = {
  minConf: 80,
  gateMinN: 5, // don't trip a hold on fewer than 5 results in the window (one loss at ~90¢ would otherwise trip it)
  gates: {
    // mid-market forks: hold when trailing 1h avg <= -2¢, release when trailing 30m avg >= releaseCents.
    // The video says "positive 25 cents ... I think". +25¢ average per contract is almost unreachable next to a +15/+20¢
    // take-profit, so 2.5 is used. Set this to 25 to follow the transcript literally.
    mid: { holdSpanMin: 60, holdBelow: -2, releaseSpanMin: 30, releaseAtLeast: 2.5 },
    // end-of-market: hold when trailing 2h avg <= -2¢, release when trailing 1h avg is positive.
    end: { holdSpanMin: 120, holdBelow: -2, releaseSpanMin: 60, releaseAtLeast: 0.0001 },
  },
  // stop/take-profit optimizer grid (transcript: SL as a % of entry, TP in cents, "off"/"hold" = none)
  // includes 0.75 because the reference bot's optimizer landed on "75% of entry" for end-of-market
  slGrid: [null, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
  // A stop is a "sell at the best price above 1¢" button press, so it fills at whatever the bid is when you press,
  // not at the stop price. The replay fills at the sampled bid this many samples (pathSampleMs each) AFTER the trigger
  // sample; 0 = the very sample that crossed the stop, 1 = ~5s of reaction/latency.
  exitLagSamples: 0,
  tpGrid: [null, 5, 10, 15, 20, 30],
  optimizerMinTrades: 8,
  pathSampleMs: 5000,
};

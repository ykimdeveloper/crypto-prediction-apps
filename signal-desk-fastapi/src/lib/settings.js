// Paper-trading settings (persisted in this browser).
//   feeCents   flat Kalshi-style taker fee per contract, charged on entry (the app assumes you buy at the ask, i.e. cross the
//              spread). A stop-loss exit is a market-style sell and is charged again; a take-profit is a resting limit order
//              (no fee) and settlement is free. Set 0 to model limit-order fills that rest and get filled. The reference
//              videos show ~0.7-0.85¢ on 88-94¢ fills; the real formula isn't confirmed here, so it is a setting.
//   midEnabled false pauses the two mid-market forks (no recs, no paper trades, no alerts) — the reference bot has them
//              paused while it is being re-tested.
const KEY = 'sd.settings';
export const DEFAULT_SETTINGS = { feeCents: 0.8, midEnabled: true };

export function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    const fee = s && Number.isFinite(+s.feeCents) ? Math.min(10, Math.max(0, +s.feeCents)) : DEFAULT_SETTINGS.feeCents;
    return { ...DEFAULT_SETTINGS, ...(s || {}), feeCents: fee };
  } catch { return DEFAULT_SETTINGS; }
}
export function saveSettings(s) {
  try { localStorage.setItem(KEY, JSON.stringify(s)); } catch { /* storage unavailable */ }
}

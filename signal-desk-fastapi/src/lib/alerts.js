// Alerts fire when a new, visible BUY rec opens (gated/shadow recs and HOLD/WAIT never alert).
// Exit alerts fire once per open rec when its bid reaches the take-profit level or falls to the stop level. Stops are
// manual on Kalshi's 15-minute markets, so this is the cue to press SELL (which fills at the best bid, not the stop price).
const KEY = 'sd.alerts';
// exits.stopPct = stop level as % of the entry price (75 = the reference bot's suggestion for end-of-market);
// exits.tpEnd    = end-of-market take-profit price in cents (mid-market forks use their own +15/+20¢ target).
export const DEFAULT_ALERTS = {
  enabled: false, mode: 'both', muted: [], strats: { rev: true, mom: true, end: true },
  exits: { enabled: true, stopPct: 75, tpEnd: 99 },
};

export function loadAlerts() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    return s ? { ...DEFAULT_ALERTS, ...s, strats: { ...DEFAULT_ALERTS.strats, ...s.strats }, exits: { ...DEFAULT_ALERTS.exits, ...s.exits } } : DEFAULT_ALERTS;
  } catch { return DEFAULT_ALERTS; }
}
export function saveAlerts(a) {
  try { localStorage.setItem(KEY, JSON.stringify(a)); } catch { /* storage unavailable */ }
}

// muted = tickers switched off; anything not muted (including tickers added later) alerts
export const shouldAlert = (a, rec) => a.enabled && a.strats[rec.stratKey] !== false && !a.muted.includes(rec.sym);
export const shouldAlertExit = (a, rec) => shouldAlert(a, rec) && !!(a.exits && a.exits.enabled);

let ctx = null;
export function unlockAudio() {
  const AC = typeof window !== 'undefined' && (window.AudioContext || window.webkitAudioContext);
  if (!AC) return null;
  if (!ctx) ctx = new AC();
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}
function tone(c, freq, at, type = 'sine') {
  const o = c.createOscillator();
  const g = c.createGain();
  o.type = type;
  o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, at);
  g.gain.exponentialRampToValueAtTime(0.3, at + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, at + 0.7);
  o.connect(g);
  g.connect(c.destination);
  o.start(at);
  o.stop(at + 0.75);
}
// two-note chime: rising for Buy Up, falling for Buy Down; three rising notes for take-profit; three low falling
// square-wave notes for a stop level (deliberately harsher so it can't be mistaken for an entry)
const CHIMES = {
  up: { notes: [[784, 0], [1175, 0.18]] },
  down: { notes: [[1175, 0], [784, 0.18]] },
  tp: { notes: [[659, 0], [880, 0.14], [1319, 0.28]] },
  stop: { notes: [[392, 0], [311, 0.14], [247, 0.28]], type: 'square' },
};
export function bell(kind) {
  const c = unlockAudio();
  if (!c) return;
  const ch = CHIMES[kind] || CHIMES.down;
  ch.notes.forEach(([f, dt]) => tone(c, f, c.currentTime + dt, ch.type));
}

export const canSpeak = () => typeof window !== 'undefined' && 'speechSynthesis' in window;
export function speak(text) {
  if (!canSpeak()) return;
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  window.speechSynthesis.speak(u);
}

export function fireAlert(a, rec) {
  const dir = rec.signal.endsWith('UP') ? 'up' : 'down';
  const voice = a.mode !== 'bell' && canSpeak();
  if (a.mode !== 'voice' || !voice) bell(dir); // bell when chosen, or as the fallback if speech isn't available
  if (voice) setTimeout(() => speak(`${rec.sym} buy ${dir}`), a.mode === 'both' ? 450 : 0);
}

// rec.kind: 'stop' (bid fell to the stop level) or 'tp' (bid reached the take-profit level)
export function fireExitAlert(a, rec) {
  const kind = rec.kind === 'stop' ? 'stop' : 'tp';
  const voice = a.mode !== 'bell' && canSpeak();
  if (a.mode !== 'voice' || !voice) bell(kind);
  if (voice) setTimeout(() => speak(`${rec.sym} ${kind === 'stop' ? 'stop level' : 'take profit'}`), a.mode === 'both' ? 600 : 0);
}

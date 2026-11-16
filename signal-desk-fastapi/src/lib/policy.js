// Live signal policy (server-side, quant15): 'original' = the algorithm untouched, 'gated' = a BUY is kept only when the trained
// 15-minute model backs its direction. One global setting on the backend, so every open tab and the paper ledger agree.
import { apiUrl } from './backend';

async function call(path, init) {
  const r = await fetch(apiUrl(path), init);
  if (r.status === 404) throw new Error('quant15 routes are not mounted — check that backend/app/main.py includes the quant15 router');
  if (!r.ok) throw new Error(`backend answered HTTP ${r.status}`);
  return r.json();
}

export const getPolicy = () => call('/api/quant15/policy');
export const setPolicy = (policy) => call('/api/quant15/policy', {
  method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ policy }),
});

// Why "gated" would not be doing anything useful right now, or null. Shown next to the switch so it never looks active when it is not.
export function policyWarning(st) {
  if (!st || st.policy !== 'gated') return null;
  if (!st.trained || st.trained.length === 0) return 'No model trained yet — signals pass through unchanged.';
  const c = st.compare;
  if (c && c.combine === false) return `Last compare did not recommend gating: ${(c.reasons && c.reasons[0]) || 'no measurable edge'}`;
  if (!c) return 'No compare run yet — gating is unproven on your data.';
  return null;
}

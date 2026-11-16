"""Original algorithm vs model vs both, scored on the SAME out-of-sample windows.

Policies
  original    every signal the replay produced (rev / mom / end)
  gated       an original signal is kept only if the 15m model's prior for that window agrees with its direction (by >= margin)
  model_only  at window start, buy the model's side when |p - 0.5| >= margin, at the neutral modeled price (50.5c), hold to settle

What is real and what is modeled: `hit` is the real settlement outcome. Cents are paper-style PnL at the *modeled* book price
(strategy.book_model), so treat cents as a consistency check, not as money.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

ENTRY_MODEL_ONLY = 50.5      # modeled ask for either side at the start of a window (50c + half spread)
MIN_TRADES = 30              # below this a "verdict" is anecdote
MIN_AUC = 0.52


def bootstrap_ci(x: np.ndarray, n: int = 1000, alpha: float = 0.10, seed: int = 0) -> tuple[float, float] | None:
    x = np.asarray(x, float)
    if len(x) < 10:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def diff_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, alpha: float = 0.10, seed: int = 1) -> tuple[float, float] | None:
    """Bootstrap interval for mean(a) - mean(b), independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 10 or len(b) < 10:
        return None
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(axis=1) - rng.choice(b, (n, len(b))).mean(axis=1)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def auc_lower_bound(auc: float, n: int, up_share: float = 0.5, z: float = 1.645) -> float:
    """One-sided 95% lower bound for an AUC (Hanley-McNeil standard error). At a few hundred labels an AUC of 0.52 is noise."""
    n1, n0 = max(n * up_share, 2), max(n * (1 - up_share), 2)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    return float(auc - z * se)


def stats(pnl: list[float], hits: list[bool]) -> dict:
    if not pnl:
        return dict(n=0, hit=None, avg_cents=None, total_cents=0.0, ci90=None)
    a = np.asarray(pnl, float)
    return dict(n=int(a.size), hit=float(np.mean(hits)), avg_cents=float(a.mean()), total_cents=float(a.sum()), ci90=bootstrap_ci(a))


def agrees(direction: str, p_up: float, margin: float) -> bool:
    return p_up >= 0.5 + margin if direction == "UP" else p_up <= 0.5 - margin


def evaluate_policies(signals: list[dict], windows: list[dict], prior: dict[int, float], margin: float = 0.02) -> dict:
    """signals/windows from replay (test windows only); prior: win_start -> P(window settles UP) from the 15m model."""
    sig = [s for s in signals if s["win_start"] in prior]
    kept = [s for s in sig if agrees(s["dir"], prior[s["win_start"]], margin)]
    vetoed = [s for s in sig if not agrees(s["dir"], prior[s["win_start"]], margin)]
    mo_pnl, mo_hit = [], []
    for w in windows:
        p = prior.get(w["win_start"])
        if p is None or abs(p - 0.5) < margin:
            continue
        side_up = p > 0.5
        won = w["up"] == side_up
        mo_hit.append(bool(won))
        mo_pnl.append(100 - ENTRY_MODEL_ONLY if won else -ENTRY_MODEL_ONLY)
    by_strategy = {}
    for s in sorted({x["strat"] for x in sig}):
        a = [x for x in sig if x["strat"] == s]
        g = [x for x in kept if x["strat"] == s]
        by_strategy[s] = dict(original=stats([x["pnl"] for x in a], [x["win"] for x in a]),
                              gated=stats([x["pnl"] for x in g], [x["win"] for x in g]))
    kept_ids = {id(x) for x in kept}
    sep = diff_ci(np.array([x["pnl"] for x in sig if id(x) in kept_ids]), np.array([x["pnl"] for x in vetoed]))
    return dict(margin=margin, kept_vs_vetoed_ci90=sep, n_windows=len([w for w in windows if w["win_start"] in prior]),
                original=stats([s["pnl"] for s in sig], [s["win"] for s in sig]),
                gated=stats([s["pnl"] for s in kept], [s["win"] for s in kept]),
                vetoed=stats([s["pnl"] for s in vetoed], [s["win"] for s in vetoed]),   # what the gate threw away
                model_only=stats(mo_pnl, mo_hit), by_strategy=by_strategy)


def verdict(pol: dict, model_auc: float | None, n_labels: int | None = None, up_share: float = 0.5) -> dict:
    """'combine' only if (1) the model's out-of-fold AUC is distinguishable from 0.5, (2) enough gated trades, (3) gating raised avg cents
    and hit rate, and (4) trades the model agreed with beat the ones it disagreed with by more than bootstrap noise (90% interval > 0)."""
    o, g, ci = pol["original"], pol["gated"], pol.get("kept_vs_vetoed_ci90")
    reasons = []
    if model_auc is None or model_auc < MIN_AUC:
        reasons.append(f"model out-of-fold AUC {None if model_auc is None else round(model_auc, 3)} < {MIN_AUC}: no measurable direction edge to gate with")
    elif n_labels and auc_lower_bound(model_auc, n_labels, up_share) <= 0.5:
        reasons.append(f"AUC {model_auc:.3f} on {n_labels} labels is within noise of 0.5 (95% lower bound {auc_lower_bound(model_auc, n_labels, up_share):.3f})")
    if g["n"] < MIN_TRADES:
        reasons.append(f"only {g['n']} gated trades (< {MIN_TRADES}): too few to judge")
    if g["n"] and o["n"] and not (g["avg_cents"] > o["avg_cents"]):
        reasons.append("gating did not raise average cents per trade")
    if g["n"] and o["n"] and not (g["hit"] >= o["hit"]):
        reasons.append("gating did not raise the settlement hit rate")
    if ci is None or ci[0] <= 0:
        reasons.append("trades the model agreed with did not beat the ones it disagreed with beyond noise"
                       + ("" if ci is None else f" (kept - vetoed 90% interval {ci[0]:.1f}..{ci[1]:.1f}c)"))
    if reasons:
        return dict(policy="original", combine=False, reasons=reasons)
    return dict(policy="gated", combine=True,
                reasons=[f"gated {g['hit']:.1%} hit / {g['avg_cents']:.1f}c vs original {o['hit']:.1%} / {o['avg_cents']:.1f}c on {g['n']} vs {o['n']} trades; "
                         f"kept - vetoed = {ci[0]:.1f}..{ci[1]:.1f}c (90%), out-of-sample"])


# ---------------- does our sigma price the settlement better than the ATR-based book? ----------------
def _p(diff: np.ndarray, sigma_total: np.ndarray) -> np.ndarray:
    return np.clip(ndtr(diff / np.maximum(sigma_total, 1e-12)), 1e-4, 1 - 1e-4)


def _scores(p: np.ndarray, up: np.ndarray) -> tuple[float, float]:
    y = up.astype(float)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigma_calibration(steps: pd.DataFrame, sig15: dict[int, float], grid=np.arange(0.6, 2.31, 0.1)) -> dict | None:
    """P(window settles UP) at each minute = Phi((price - strike) / (sigma_1m * sqrt(minutes left) * s)), scored against the real outcome.
    Each source gets its own scale s tuned on the earlier half of the steps, then is scored on the later half. Also shows the live default
    (ATR x 1.4) untouched. `sig15`: window start -> HAR per-15m-bar sigma known BEFORE the window (log-return units)."""
    s = steps.dropna(subset=["atr"]).copy()
    s = s[(s["m"] >= 1) & (s["m"] <= 13) & s["win_start"].isin(sig15.keys())]
    if len(s) < 200:
        return None
    s["rem"] = 15 - s["m"]
    s["diff"] = s["price"] - s["strike"]
    s["sig_har"] = s["win_start"].map(sig15) / np.sqrt(15.0) * s["price"]          # per-minute price sigma
    cut = s["win_start"].quantile(0.5)
    fit, ev = s[s["win_start"] < cut], s[s["win_start"] >= cut]
    if len(fit) < 100 or len(ev) < 100:
        return None
    out = []
    live_p = ev["p_book"].to_numpy()
    b, l = _scores(np.clip(live_p, 1e-4, 1 - 1e-4), ev["up"].to_numpy())
    out.append(dict(source="ATR x 1.4 (what the live book uses)", scale=1.4, brier=b, logloss=l))
    for name, col in (("ATR, scale re-tuned", "atr"), ("HAR forecast, scale tuned", "sig_har")):
        def tot(d, sc, col=col):
            return d[col].to_numpy() * np.sqrt(d["rem"].to_numpy()) * sc
        best = min(grid, key=lambda sc: _scores(_p(fit["diff"].to_numpy(), tot(fit, sc)), fit["up"].to_numpy())[1])
        b, l = _scores(_p(ev["diff"].to_numpy(), tot(ev, best)), ev["up"].to_numpy())
        out.append(dict(source=name, scale=float(best), brier=b, logloss=l))
    out.append(dict(source="coin flip (0.5)", scale=None, brier=0.25, logloss=float(np.log(2))))
    return dict(n_eval_steps=int(len(ev)), n_fit_steps=int(len(fit)), rows=out)

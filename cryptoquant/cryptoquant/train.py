import argparse
import joblib
import numpy as np
import pandas as pd
from . import config, vol, models as M
from .data import get_bars
from .features import build_features, make_label


def run(symbol, refresh=False):
    bars = get_bars(symbol, refresh)
    other = get_bars(config.OTHER[symbol], refresh)
    H = config.HORIZON
    gap = pd.Timedelta(config.BAR) * H          # purge gap so labels don't leak across splits

    n = len(bars)
    t_tr, t_va = bars.index[int(n * config.TRAIN_FRAC)], bars.index[int(n * (config.TRAIN_FRAC + config.VALID_FRAC))]
    r = np.log(bars.close).diff()

    # fit the volatility models on TRAIN only
    garch = vol.fit_garch(r[r.index < t_tr])
    har = vol.fit_har(r[r.index < t_tr], H)

    X = build_features(bars, other, garch, har)
    y, fwd = make_label(bars, H)
    d = X.assign(y=y, fwd=fwd).dropna()

    tr = d[d.index < t_tr - gap]
    va = d[(d.index >= t_tr) & (d.index < t_va - gap)]
    te = d[d.index >= t_va]
    feats = list(X.columns)
    print(f"\n=== {symbol} | train {len(tr)}  valid {len(va)}  test {len(te)} | features {len(feats)}")

    ms = M.fit_all(tr[feats], tr.y, va[feats], va.y)
    probs = M.predict_all(ms, te[feats])
    probs["ensemble"] = M.ensemble({k: v for k, v in probs.items()})
    rows = {k: M.score(te.y, p) for k, p in probs.items()}
    rows["baseline"] = M.score(te.y, np.full(len(te), tr.y.mean()))
    print(pd.DataFrame(rows).T.round(4))

    # vol regimes from train HAR forecast
    q = tr["har_vol"].quantile([1 / 3, 2 / 3]).tolist()
    reg = te["har_vol"].map(lambda v: vol.regime(v, q))
    p = probs["ensemble"]
    out = []
    for name in ("low", "mid", "high"):
        m = (reg.values == name) & (np.abs(p - 0.5) >= config.EDGE[name])
        if m.sum() == 0:
            continue
        sign = np.where(p[m] > 0.5, 1, -1)
        out.append({
            "regime": name, "trades": int(m.sum()), "coverage": m.sum() / (reg.values == name).sum(),
            "hit_rate": float(((p[m] > 0.5) == (te.y.values[m] == 1)).mean()),
            "avg_ret_bps_gross": float((sign * te.fwd.values[m]).mean() * 1e4),
        })
    print("\nThresholded signal on test set (gross of fees/slippage):")
    print(pd.DataFrame(out).round(4).to_string(index=False))

    config.ART_DIR.mkdir(exist_ok=True, parents=True)
    joblib.dump({"models": ms, "features": feats, "garch": garch, "har": har, "vol_q": q},
                config.ART_DIR / f"{symbol}.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH"])
    ap.add_argument("--refresh", action="store_true", help="re-download data from CryptoDataDownload")
    a = ap.parse_args()
    for s in a.symbols:
        run(s.upper(), a.refresh)


if __name__ == "__main__":
    main()

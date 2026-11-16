import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import config, db  # noqa: E402
from app.approaches import evaluate, models, service, vol  # noqa: E402
from app.approaches.features import FEATURES, build_features  # noqa: E402
from app.approaches.labels import make_labels  # noqa: E402
from app.approaches.validation import walk_forward  # noqa: E402
from app.constants import MIN_MS  # noqa: E402

T0 = 1_790_000_000_000 // MIN_MS * MIN_MS


def synth(n=6000, seed=1, signal=True, start=100.0, holes=()):
    """1m candles. signal=True: the drift flips sign every ~150 bars (a persistent regime), so past returns predict future ones."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    if signal:
        sign, i = 1, 0
        while i < n:
            L = int(rng.integers(100, 220))
            drift[i:i + L] = sign * 2.5e-4
            sign, i = -sign, i + L
    lr = drift + rng.normal(0, 3e-4, n)
    c = start * np.exp(np.cumsum(lr))
    o = np.r_[start, c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 1e-4, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    v = rng.lognormal(0, 0.4, n)
    ts = T0 + np.arange(n) * MIN_MS
    d = pd.DataFrame(dict(ts=ts, open=o, high=hi, low=lo, close=c, volume=v))
    for a, b in holes:
        d = d.drop(index=range(a, b))
    return d.reset_index(drop=True)


class TestFeaturesLabels(unittest.TestCase):
    def test_no_lookahead(self):
        d = synth(400)
        full, cut = build_features(d), build_features(d.iloc[:300].copy())
        pd.testing.assert_frame_equal(full.iloc[:300].reset_index(drop=True), cut.reset_index(drop=True))

    def test_gap_blanks_features_and_labels(self):
        d = synth(400, holes=[(200, 210)])
        f = build_features(d)
        self.assertTrue(f.iloc[200:260].isna().all(axis=1).all())      # 60 bars after the hole are unusable
        sig = vol.ewma(vol.log_returns(d["close"]))
        lab = make_labels(d, sig, 15, 0.0, 0.0)
        self.assertTrue(lab["fwd"].iloc[186:200].isna().all())           # labels whose forward window spans the hole
        self.assertTrue(lab["fwd"].iloc[100:150].notna().all())

    def test_neutral_zone(self):
        d = synth(400, signal=False)
        sig = vol.ewma(vol.log_returns(d["close"]))
        lab = make_labels(d, sig, 15, cost_bps=8, k=0.5)
        m = lab["fwd"].notna() & lab["theta"].notna()
        self.assertTrue(lab.loc[m & (lab["fwd"].abs() <= lab["theta"]), "y"].isna().all())
        self.assertTrue((lab.loc[lab["y"] == 1, "fwd"] > lab.loc[lab["y"] == 1, "theta"]).all())


class TestValidation(unittest.TestCase):
    def test_purge_and_order(self):
        ts = T0 + np.arange(2000) * MIN_MS
        prev_end = None
        for j, tr, te in walk_forward(ts, 15, MIN_MS, 4):
            start = ts[te].min()
            self.assertTrue((ts[tr] + 15 * MIN_MS <= start).all())     # no training label reaches into the test block
            self.assertFalse((tr & te).any())
            self.assertGreaterEqual(tr.sum(), 900)
            if prev_end is not None:
                self.assertEqual(start, prev_end)                       # blocks are contiguous and disjoint
            prev_end = ts[te].max() + MIN_MS
        self.assertEqual(j, 3)

    def test_pooled_symbols_split_by_time(self):
        ts = np.repeat(T0 + np.arange(1000) * MIN_MS, 3)
        for _, tr, te in walk_forward(ts, 15, MIN_MS, 3):
            self.assertLess(ts[tr].max(), ts[te].min())


class TestCompare(unittest.TestCase):
    def run_compare(self, signal, seed):
        frames = {"A": synth(5000, seed, signal), "B": synth(5000, seed + 10, signal)}
        data = evaluate.build_dataset(frames, 15, 8.0, 0.5)
        return evaluate.compare(data, 15, 8.0, ["momentum", "ema_cross", "logreg", "hist_gb", "lightgbm"], n_folds=3)

    def test_finds_planted_signal(self):
        r = {x["approach"]: x for x in self.run_compare(True, 1)["results"]}
        self.assertGreater(r["hist_gb"]["auc"], 0.65)
        self.assertGreater(r["logreg"]["auc"], 0.65)
        self.assertGreater(r["hist_gb"]["trading"]["net_bps_total"], 0)
        if not models.LightGBM.available():
            self.assertFalse(r["lightgbm"]["available"])                # missing optional lib is reported, not fatal
        self.assertIn("ensemble_avg", r)

    def test_no_edge_on_noise(self):
        out = self.run_compare(False, 2)
        for x in out["results"]:
            if x["auc"] is not None:
                self.assertLess(x["auc"], 0.6, x["approach"])
        for x in out["results"]:
            if x["approach"] in ("hist_gb", "logreg"):
                self.assertLessEqual(x["trading"]["net_bps_total"], 0.0 + 1e-9 if x["trading"]["n_trades"] else 1)

    def test_too_little_data(self):
        data = evaluate.build_dataset({"A": synth(150)}, 15, 8.0, 0.5)
        with self.assertRaises(ValueError):
            evaluate.compare(data, 15, 8.0)


class TestVolAndDecision(unittest.TestCase):
    def test_vol_methods_track_truth(self):
        d = synth(4000, signal=False)
        res = {r["method"]: r for r in vol.score_vol({"A": d}, 15)}
        self.assertGreater(res["ewma"]["corr"] + 1, 0)                   # runs
        self.assertLess(res["har"]["mae_bps_per_bar"], 3.0)              # true per-bar sigma is 3 bps
        if not vol.garch_available():
            self.assertEqual(res["garch"]["note"], "arch not installed")

    def test_decide(self):
        q = service.decide(0.51, 3e-4, 15, 8, 0.55, 1.5)
        self.assertEqual(q["trade_action"], "no_trade")
        quiet = service.decide(0.70, 1e-4, 15, 8, 0.55, 1.5)             # confident but the tape is dead
        self.assertEqual(quiet["trade_action"], "no_trade"); self.assertIn("too quiet", quiet["reason"])
        go = service.decide(0.72, 6e-4, 15, 8, 0.55, 1.5)
        self.assertEqual(go["trade_action"], "long")
        self.assertEqual(service.decide(0.28, 6e-4, 15, 8, 0.55, 1.5)["trade_action"], "short")


class TestService(unittest.TestCase):
    def test_train_predict_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config.MODEL_DIR = Path(tmp)
            db.init(":memory:")
            for sym, seed in (("BTC/USDT", 5), ("ETH/USDT", 6)):
                d = synth(4000, seed)
                db.upsert_candles("x", sym, "1m", [dict(t=int(r.ts), o=r.open, h=r.high, l=r.low, c=r.close, v=r.volume) for r in d.itertuples()])
            meta = service.train_final("hist_gb", "x", ["BTC/USDT", "ETH/USDT"])
            self.assertTrue(meta["calibrated"]); self.assertGreater(meta["holdout_auc"], 0.6)
            out = service.predict("x", "BTC/USDT")
            self.assertIn(out["trade_action"], ("long", "short", "no_trade"))
            self.assertTrue(0 <= out["direction_probability_up"] <= 1)
            self.assertEqual(len(service.saved_models()), 1)
            with self.assertRaises(LookupError):
                service.predict("x", "BTC/USDT", "catboost")


if __name__ == "__main__":
    unittest.main()

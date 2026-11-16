import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

from app import config, db  # noqa: E402
from app.constants import CONTRACT_MS, MIN_MS  # noqa: E402
from app.quant15 import bars as B, combine, model15 as M, replay as R, service, vol15 as V  # noqa: E402
from app.quant15.features15 import FEATURES15, build_features15  # noqa: E402

T0 = (1_790_000_000_000 // CONTRACT_MS) * CONTRACT_MS      # aligned to a contract window


def synth1m(n=30_000, seed=3, drift=5e-5, vol=3e-4, regime=(600, 1500), start=100.0, holes=()):
    """1m candles whose drift flips sign every 10-25 hours: a persistent regime the 15m model can learn."""
    rng = np.random.default_rng(seed)
    d = np.zeros(n)
    sign, i = 1, 0
    while i < n:
        L = int(rng.integers(*regime))
        d[i:i + L] = sign * drift
        sign, i = -sign, i + L
    lr = d + rng.normal(0, vol, n)
    c = start * np.exp(np.cumsum(lr))
    o = np.r_[start, c[:-1]]
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 1e-4, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    df = pd.DataFrame(dict(ts=T0 + np.arange(n) * MIN_MS, open=o, high=hi, low=lo, close=c, volume=rng.lognormal(0, 0.4, n)))
    for a, b in holes:
        df = df.drop(index=range(a, b))
    return df.reset_index(drop=True)


def store(exchange, symbol, d1, tf="1m"):
    db.upsert_candles(exchange, symbol, tf, [dict(t=int(r.ts), o=r.open, h=r.high, l=r.low, c=r.close, v=r.volume) for r in d1.itertuples()])


class TestBars(unittest.TestCase):
    def test_resample_alignment_and_completeness(self):
        d = synth1m(300, holes=[(40, 46)])            # 6 minutes missing inside window 2 (minutes 30..44) -> 9 left, dropped
        b = B.resample_15m(d)
        self.assertTrue((b["ts"] % CONTRACT_MS == 0).all())
        self.assertNotIn(T0 + 2 * CONTRACT_MS, set(b["ts"]))
        w = d[(d["ts"] >= T0) & (d["ts"] < T0 + CONTRACT_MS)]
        r = b[b["ts"] == T0].iloc[0]
        self.assertAlmostEqual(r["open"], w["open"].iloc[0]); self.assertAlmostEqual(r["close"], w["close"].iloc[-1])
        self.assertAlmostEqual(r["high"], w["high"].max()); self.assertAlmostEqual(r["low"], w["low"].min())
        self.assertAlmostEqual(r["volume"], w["volume"].sum())

    def test_backfill_paginates_and_resumes(self):
        db.init(":memory:")
        try:
            class Fake:
                calls = 0
                def parse_timeframe(self, tf): return 60
                def milliseconds(self): return T0 + 2500 * MIN_MS
                def fetch_ohlcv(self, sym, tf, since=None, limit=1000):
                    Fake.calls += 1
                    ts = range(((since - T0) // MIN_MS) * MIN_MS + T0 if since >= T0 else T0, T0 + 2500 * MIN_MS, MIN_MS)
                    return [[t, 1, 2, 0.5, 1.5, 3] for t in ts if t >= since][:limit]
            ex = Fake()
            with mock.patch.object(Fake, "milliseconds", lambda s: T0 + 2500 * MIN_MS):
                r = B.backfill("x", "BTC/USDT", days=1, tf="1m", ex=ex, page=1000)
                self.assertEqual(len(db.dataset("x", "BTC/USDT", "1m")), 1440)      # exactly one day, no duplicates
                self.assertGreaterEqual(Fake.calls, 2)
                n = B.backfill("x", "BTC/USDT", days=1, tf="1m", ex=ex, page=1000)["fetched"]
                self.assertEqual(n, 0)                                                # resumes after the newest candle
        finally:
            db.close()

    def test_backfill_fills_history_before_live_recorded_candles(self):
        """A few hours recorded live must not stop a 1-day backfill from fetching the earlier hours; Coinbase-style page cap respected."""
        db.init(":memory:")
        try:
            END = T0 + 2500 * MIN_MS

            class Fake:
                max_limit = 0
                def parse_timeframe(self, tf): return 60
                def milliseconds(self): return END
                def fetch_ohlcv(self, sym, tf, since=None, limit=1000):
                    Fake.max_limit = max(Fake.max_limit, limit)
                    if limit > 300:
                        raise RuntimeError("limit must be <= 300")
                    return [[t, 1, 2, 0.5, 1.5, 3] for t in range(((since + 59_999) // MIN_MS) * MIN_MS, END, MIN_MS)][:limit]
            db.upsert_candles("coinbase", "BTC/USDT", "1m",
                              [dict(t=END - (i + 1) * MIN_MS, o=1, h=2, l=.5, c=1.5, v=1) for i in range(200)])   # only the last 200 minutes
            r = B.backfill("coinbase", "BTC/USDT", days=1, tf="1m", ex=Fake(), page=1000)
            self.assertEqual(len(db.dataset("coinbase", "BTC/USDT", "1m")), 1440)
            self.assertLessEqual(Fake.max_limit, 300)
            self.assertGreater(r["fetched"], 1000)
        finally:
            db.close()

    def test_ensure_one_ccxt_path_returns_dict(self):
        """Regression: _ensure_one passed 'fetched' twice and raised TypeError for every non-binance exchange."""
        try:
            from app.quant15 import routes
        except ModuleNotFoundError as e:   # fastapi not installed in this environment
            self.skipTest(str(e))
        db.init(":memory:")
        try:
            with mock.patch.object(B, "backfill", return_value=dict(exchange="coinbase", symbol="BTC/USDT", tf="1m", fetched=123, from_ts=1, to_ts=2)):
                r = routes._ensure_one("coinbase", "BTC/USDT", 45)
            self.assertEqual(r["source"], "ccxt"); self.assertEqual(r["fetched"], 123); self.assertIn("windows", r)
        finally:
            db.close()

    def test_inventory_and_sparse_pair_error(self):
        """A thin pair (few 1m candles) shows can_train False in the inventory and a train error that names the stored counts."""
        from app.quant15 import service as S
        db.init(":memory:")
        try:
            db.upsert_candles("coinbase", "BTC/USDT", "1m", [dict(t=T0 + i * 7 * MIN_MS, o=1, h=2, l=.5, c=1.5, v=1) for i in range(300)])   # 1 candle / 7 min
            db.upsert_candles("coinbaseexchange", "BTC/USD", "1m", [dict(t=T0 + i * MIN_MS, o=1, h=2, l=.5, c=1.5, v=1) for i in range(900 * 3)])
            inv = {(r["exchange"], r["symbol"]): r for r in S.inventory()["rows"]}
            thin, full = inv[("coinbase", "BTC/USDT")], inv[("coinbaseexchange", "BTC/USD")]
            self.assertEqual(thin["usable_15m_bars"], 0); self.assertFalse(thin["can_train"]); self.assertLess(thin["coverage"], 0.2)
            self.assertEqual(full["coverage"], 1.0); self.assertGreater(full["usable_15m_bars"], 0)
            with self.assertRaises(ValueError) as cm:
                S.train_final("coinbase", ["BTC/USDT"])
            self.assertIn("candles_1m", str(cm.exception)); self.assertIn("coinbaseexchange", str(cm.exception))
        finally:
            db.close()

    def test_backfill_skips_empty_pages(self):
        """Thin pairs return empty pages for quiet stretches; that must not end the backfill (Coinbase BTC/USDT fetched 1 candle)."""
        db.init(":memory:")
        try:
            END = T0 + 2000 * MIN_MS
            LATE = T0 + 1500 * MIN_MS          # nothing trades before this minute

            class Fake:
                def parse_timeframe(self, tf): return 60
                def milliseconds(self): return END
                def fetch_ohlcv(self, sym, tf, since=None, limit=300):
                    lo = ((since + 59_999) // MIN_MS) * MIN_MS
                    return [[t, 1, 2, .5, 1.5, 3] for t in range(lo, min(lo + limit * MIN_MS, END), MIN_MS) if t >= LATE]
            r = B.backfill("x", "BTC/USDT", days=2, tf="1m", ex=Fake(), page=300)
            self.assertGreater(len(db.dataset("x", "BTC/USDT", "1m")), 400)
        finally:
            db.close()


class TestFeaturesAndLabels(unittest.TestCase):
    def test_no_lookahead(self):
        b = B.resample_15m(synth1m(30 * 900))
        full, cut = build_features15(b), build_features15(b.iloc[:900].copy())
        pd.testing.assert_frame_equal(full.iloc[:900].reset_index(drop=True), cut.reset_index(drop=True))

    def test_label_is_next_window_direction(self):
        b = B.resample_15m(synth1m(30 * 900))
        data = M.build_dataset15({"BTC/USDT": b})
        d = data.merge(b.assign(ts=b["ts"] - CONTRACT_MS)[["ts", "open", "close"]].rename(columns={"open": "o1", "close": "c1"}), on="ts")
        ok = d["y"].notna()
        self.assertTrue(((d.loc[ok, "c1"] >= d.loc[ok, "o1"]).astype(float) == d.loc[ok, "y"]).all())
        self.assertTrue(data["y"].iloc[-1:].isna().all())       # the newest bar has no future window
        self.assertEqual(list(data[FEATURES15].columns), FEATURES15)


class TestReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d1 = synth1m(6000, seed=5, drift=2.5e-4, vol=3e-4, regime=(120, 260))
        cls.rp = R.replay(cls.d1)

    def test_windows_and_steps(self):
        rp = self.rp
        self.assertGreater(len(rp["windows"]), 300)
        for w in rp["windows"][:50]:
            self.assertEqual(w["up"], w["close_end"] > w["strike"])
        self.assertTrue(rp["steps"]["m"].between(1, 15).all())

    def test_signal_records_follow_paper_rules(self):
        sigs = self.rp["signals"]
        self.assertGreater(len(sigs), 5, "planted-trend data should trigger some rev/mom/end entries")
        seen = set()
        for s in sigs:
            k = (s["win_start"], s["strat"]); self.assertNotIn(k, seen); seen.add(k)
            self.assertGreaterEqual(s["conf"], 80)
            won = s["close_end"] > s["strike"] if s["dir"] == "UP" else s["close_end"] < s["strike"]
            self.assertEqual(s["win"], won)
            self.assertAlmostEqual(s["settle_pnl"], 100 - s["entry"] if won else -s["entry"])
            self.assertAlmostEqual(s["pnl"], s["tp"] if s["how"] == "tp" else s["settle_pnl"])
            if s["strat"] == "end":
                self.assertIsNone(s["tp"])

    def test_signals_now_matches_replay_step(self):
        d = self.d1.iloc[:3000 + 8].reset_index(drop=True)     # stop 8 minutes into a window (3000 % 15 == 0)
        sig = R.signals_now(d)
        self.assertIn("mom", sig); self.assertAlmostEqual(sig["remMin"], 7.0)     # 8 candles closed -> 7 minutes left


class TestCombine(unittest.TestCase):
    def _sigs(self, n, hit_agree, hit_disagree):
        rng = np.random.default_rng(1)
        sigs, prior = [], {}
        for i in range(n):
            w = 1000 + i
            agree = i % 2 == 0
            hit = rng.random() < (hit_agree if agree else hit_disagree)
            prior[w] = 0.6 if agree else 0.4          # signals are all UP: 0.6 agrees, 0.4 disagrees
            sigs.append(dict(win_start=w, strat="mom", dir="UP", win=bool(hit), pnl=(50.0 if hit else -50.0)))
        return sigs, prior

    def test_gate_helps_when_model_is_right(self):
        sigs, prior = self._sigs(400, 0.8, 0.3)
        pol = combine.evaluate_policies(sigs, [], prior, 0.02)
        self.assertEqual(pol["gated"]["n"], 200); self.assertGreater(pol["gated"]["avg_cents"], pol["original"]["avg_cents"])
        v = combine.verdict(pol, 0.56, n_labels=2000)
        self.assertTrue(v["combine"]); self.assertEqual(v["policy"], "gated")

    def test_verdict_refuses_without_evidence(self):
        sigs, prior = self._sigs(400, 0.8, 0.3)
        pol = combine.evaluate_policies(sigs, [], prior, 0.02)
        self.assertFalse(combine.verdict(pol, 0.505)["combine"])              # model has no edge
        few = combine.evaluate_policies(sigs[:20], [], prior, 0.02)
        self.assertFalse(combine.verdict(few, 0.6)["combine"])                # too few trades
        self.assertFalse(combine.verdict(pol, 0.53, n_labels=150)["combine"])  # AUC 0.53 on 150 labels is noise
        self.assertGreater(combine.auc_lower_bound(0.60, 2000), 0.5); self.assertLess(combine.auc_lower_bound(0.53, 150), 0.5)
        sigs, prior = self._sigs(400, 0.4, 0.6)
        self.assertFalse(combine.verdict(combine.evaluate_policies(sigs, [], prior, 0.02), 0.6)["combine"])  # gate makes it worse

    def test_model_only_pays_50_5(self):
        wins = [dict(win_start=w, up=True) for w in range(100)]
        pol = combine.evaluate_policies([], wins, {w: 0.7 for w in range(100)}, 0.02)
        self.assertAlmostEqual(pol["model_only"]["avg_cents"], 100 - combine.ENTRY_MODEL_ONLY)

    def test_sigma_calibration_finds_the_true_scale(self):
        rng = np.random.default_rng(0)
        rows, sig15 = [], {}
        s0, price = 1.0, 100.0                                        # true per-minute sigma in price units
        for w in range(400):
            sig15[w] = s0 * np.sqrt(15) / price
            path = np.cumsum(rng.normal(0, s0, 15))
            for m in range(1, 14):
                diff = path[m - 1]
                up = path[-1] > 0
                atr = 0.5                                              # deliberately mis-scaled ATR
                p_book = float(np.clip(ndtr(diff / (atr * np.sqrt(15 - m) * 1.4)), 1e-4, 1 - 1e-4))
                rows.append((w, m, price + diff, price, atr, p_book, up))
        steps = pd.DataFrame(rows, columns=R.STEP_COLS)
        out = combine.sigma_calibration(steps, sig15)
        by = {r["source"]: r for r in out["rows"]}
        har = by["HAR forecast, scale tuned"]
        self.assertAlmostEqual(har["scale"], 1.0, delta=0.2)
        self.assertLess(har["logloss"], by["ATR x 1.4 (what the live book uses)"]["logloss"])
        self.assertLess(har["logloss"], np.log(2))


class TestEndToEnd(unittest.TestCase):
    EX, SYM = "test", "BTC/USDT"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls._old = config.MODEL_DIR
        config.MODEL_DIR = Path(cls.tmp.name)
        db.init(":memory:")
        cls.d1 = synth1m(30_000, seed=11, drift=8e-5, regime=(600, 1500))
        store(cls.EX, cls.SYM, cls.d1)

    @classmethod
    def tearDownClass(cls):
        config.MODEL_DIR = cls._old
        db.close(); cls.tmp.cleanup()

    def test_1_compare_report(self):
        t = time.time()
        rep = service.run_compare(self.EX, [self.SYM], approaches=["hist_gb"], n_folds=3)
        print(f"\n[compare took {time.time() - t:.1f}s]")
        ens = next(r for r in rep["model"] if r["approach"] == "ensemble")
        self.assertGreater(ens["auc"], 0.53, "planted persistent-drift regime should be learnable at 15m")
        self.assertIsNotNone(rep["policies"]); self.assertGreater(rep["policies"]["original"]["n"], 0)
        self.assertIn(rep["verdict"]["policy"], ("original", "gated"))
        self.assertIsNotNone(rep["sigma_calibration"])
        self.assertTrue((config.MODEL_DIR / "quant15_policy.json").exists())
        print(rep["verdict"], {k: rep["policies"][k]["n"] for k in ("original", "gated", "model_only")})

    def test_2_train_and_decide(self):
        meta = service.train_final(self.EX, [self.SYM], approaches=["hist_gb"])
        self.assertTrue(meta["calibrated"])
        now = int(self.d1["ts"].iloc[-1]) + MIN_MS
        # cut the stored data 8 minutes into a window so the "live" window is in progress
        W = (now // CONTRACT_MS) * CONTRACT_MS
        bundle = service.load(service._bundle_path(self.EX))
        pm = service.prior_now(bundle, self.EX, self.SYM, now_ms=W + 8 * MIN_MS)
        self.assertTrue(pm["available"], pm); self.assertGreater(pm["p_up"], 0.0); self.assertIn(pm["regime"], ("low", "mid", "high"))
        # too little history for the live window -> unavailable, not a crash
        self.assertFalse(service.prior_now(bundle, self.EX, self.SYM, now_ms=T0 + 100 * MIN_MS)["available"])

    def test_3_gate_vetoes_only_on_disagreement(self):
        buy = dict(signal="BUY UP", conf=90, ask=60.0, reason="x", tp=20)
        wait = dict(signal="WAIT", conf=None, reason="no")
        fake = dict(module="MID", story="Momentum Continuation", lean=0.5, remMin=8.0, book=None, rev=wait, mom=buy, end=wait)
        for p, expect in ((0.30, "WAIT"), (0.70, "BUY UP")):
            with mock.patch.object(service, "STALE_S", 10**12), mock.patch.object(service.R, "signals_now", return_value=fake), \
                 mock.patch.object(service, "prior_now", return_value=dict(available=True, p_up=p, regime="mid")):
                out = service.decide(self.EX, self.SYM, policy="gated", margin=0.02)
            self.assertEqual(out["strategies"]["mom"]["final"], expect)
            self.assertEqual(out["strategies"]["rev"]["final"], "WAIT")
        with mock.patch.object(service, "STALE_S", 10**12), mock.patch.object(service.R, "signals_now", return_value=fake), \
             mock.patch.object(service, "prior_now", return_value=dict(available=True, p_up=0.3, regime="mid")):
            out = service.decide(self.EX, self.SYM, policy="original")
        self.assertEqual(out["strategies"]["mom"]["final"], "BUY UP")          # default path never changes the original
        with mock.patch.object(service, "STALE_S", -10**12), mock.patch.object(service.R, "signals_now", return_value=fake):   # force "stale"
            out = service.decide(self.EX, self.SYM, policy="original")
        self.assertEqual(out["strategies"]["mom"]["final"], "WAIT"); self.assertIn("stale", out["strategies"]["mom"]["note"])
        with mock.patch.object(service.R, "signals_now", return_value=fake), \
             mock.patch.object(service, "prior_now", return_value=dict(available=False, reason="x")), \
             mock.patch.object(service, "STALE_S", 10**12):
            out = service.decide(self.EX, self.SYM, policy="gated")
        self.assertEqual(out["strategies"]["mom"]["final"], "BUY UP")          # model missing -> pass through, flagged
        self.assertIn("unavailable", out["strategies"]["mom"]["note"])


class TestNoiseGetsNoCombine(unittest.TestCase):
    def test_pure_noise_keeps_the_original(self):
        tmp = tempfile.TemporaryDirectory(); old = config.MODEL_DIR; config.MODEL_DIR = Path(tmp.name)
        db.init(":memory:")
        try:
            store("noise", "BTC/USDT", synth1m(30_000, seed=21, drift=0.0))
            rep = service.run_compare("noise", ["BTC/USDT"], approaches=["hist_gb"], n_folds=3)
            self.assertFalse(rep["verdict"]["combine"], rep["verdict"])
            self.assertEqual(rep["verdict"]["policy"], "original")
            self.assertTrue(rep["verdict"]["reasons"])
        finally:
            config.MODEL_DIR = old; db.close(); tmp.cleanup()


if __name__ == "__main__":
    unittest.main()

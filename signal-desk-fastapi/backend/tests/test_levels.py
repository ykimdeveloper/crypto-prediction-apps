"""Levels toolkit tests: isolation from the desk, no look-ahead, touch semantics, honest thin/gapped cells, toggle default."""
import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app.levels import candidates as C, config as LC, service as S, touches as T

APP = Path(__file__).resolve().parent.parent / "app"
T0 = 1_700_000_000_000 // 3_600_000 * 3_600_000
CFG = dict(LC.DEFAULTS, enabled=True, min_touches=0)


def synth(n=3000, tf_ms=300_000, seed=3):
    r = np.random.default_rng(seed)
    close = 100 + 5 * np.sin(np.arange(n) / 40) + np.cumsum(r.normal(0, 0.15, n))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + np.abs(r.normal(0, 0.15, n))
    lo = np.minimum(open_, close) - np.abs(r.normal(0, 0.15, n))
    return pd.DataFrame(dict(ts=T0 + np.arange(n, dtype="int64") * tf_ms, open=open_, high=hi, low=lo, close=close, volume=r.uniform(1, 10, n)))


class Isolation(unittest.TestCase):
    def test_desk_never_imports_levels(self):
        for rel in ("strategy.py", "paper.py", "constants.py", "engine.py", "hub.py", "quant15/live.py"):
            tree = ast.parse((APP / rel).read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or ""] + [a.name for a in node.names]
                elif isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                self.assertFalse(any("levels" in n.split(".") for n in names), f"{rel} imports the levels package: {names}")


class Config(unittest.TestCase):
    def test_default_off_and_persisted(self):
        self.assertFalse(LC.DEFAULTS["enabled"])
        with tempfile.TemporaryDirectory() as tmp:
            old = LC.app_config.MODEL_DIR
            LC.app_config.MODEL_DIR, LC._cfg = Path(tmp), None
            try:
                self.assertFalse(LC.get()["enabled"])
                LC.update(dict(enabled=True, spans=["1y"], windows=24))
                LC._cfg = None
                c = LC.get()
                self.assertTrue(c["enabled"])
                self.assertEqual(c["spans"], ["12mo"])          # "1 year" and "12 months" are one span
                self.assertEqual(json.loads((Path(tmp) / "levels_config.json").read_text())["windows"], 24)
                with self.assertRaises(ValueError):
                    LC.update(dict(show="bogus"))
            finally:
                LC.app_config.MODEL_DIR, LC._cfg = old, None


class NoLookahead(unittest.TestCase):
    def test_future_cannot_change_levels(self):
        d = synth()
        as_of = int(d["ts"].iloc[2200]) + 300_000
        full = S.compute_cell(d, "5m", "1w", as_of, CFG, all_touches=True)
        self.assertEqual(full["status"], "ok")
        self.assertGreater(len(full["zones"]), 0)
        cut = d[d["ts"] + 300_000 <= as_of].reset_index(drop=True)
        self.assertEqual(json.dumps(full, sort_keys=True), json.dumps(S.compute_cell(cut, "5m", "1w", as_of, CFG, all_touches=True), sort_keys=True))
        poisoned = d.copy()
        poisoned.loc[poisoned["ts"] >= as_of, ["open", "high", "low", "close"]] *= 3.0
        self.assertEqual(json.dumps(full, sort_keys=True), json.dumps(S.compute_cell(poisoned, "5m", "1w", as_of, CFG, all_touches=True), sort_keys=True))

    def test_no_touch_before_level_was_known(self):
        d = synth()
        as_of = int(d["ts"].iloc[-1]) + 300_000
        for z in S.compute_cell(d, "5m", "1w", as_of, CFG, all_touches=True)["zones"]:
            self.assertTrue(all(t["ts"] >= z["scan_from"] for t in z["touches"]))


def candles(rows, tf_ms=300_000):
    return pd.DataFrame([dict(ts=T0 + i * tf_ms, open=o, high=h, low=l, close=c, volume=1.0) for i, (o, h, l, c) in enumerate(rows)])


ZONE = dict(low=100.0, high=101.0, scan_from=T0)


def log(rows):
    d = candles(rows)
    return T.touch_log(d, ZONE, pd.Series(1.0, index=d.index), "5m", 300_000)


class TouchSemantics(unittest.TestCase):
    def test_held_from_below(self):
        t, flips = log([(95, 96, 94, 95), (95, 100.5, 96, 99), (99, 99.5, 98, 98.5), (98.5, 99, 97, 97.5), (97.5, 98, 97, 97), (97, 98, 96, 97), (97, 98, 96, 97)])
        self.assertEqual([(x["approach"], x["outcome"]) for x in t], [("from_below", "held")])
        self.assertEqual(flips, [])

    def test_broke_and_role_flip(self):
        t, flips = log([(95, 96, 94, 95), (95, 100.5, 96, 102), (102, 103, 101.5, 102.5), (102.5, 103, 102, 102.6), (102.6, 103, 102, 102.7), (102.7, 103, 102, 102.8), (102.8, 103, 102, 102.9)])
        self.assertEqual(t[0]["outcome"], "broke")
        self.assertEqual(flips[0]["from_role"], "resistance")
        self.assertEqual(flips[0]["to_role"], "support")

    def test_unclosed_horizon_is_pending_not_counted(self):
        t, _ = log([(95, 96, 94, 95), (95, 100.5, 96, 99), (99, 99.5, 98, 98.5)])
        self.assertEqual(t[0]["outcome"], "pending")
        self.assertEqual(T.summarize(t, [], T0)["held"], 0)

    def test_staying_inside_is_one_touch(self):
        t, _ = log([(95, 96, 94, 95), (95, 100.5, 96, 100.2), (100.2, 100.8, 100.1, 100.4), (100.4, 100.9, 100.0, 100.5)] + [(100.5, 100.9, 100.0, 100.5)] * 4)
        self.assertEqual(len(t), 1)


class HonestCells(unittest.TestCase):
    def test_thin_cell_says_so(self):
        d = synth(24, 3_600_000)
        c = S.compute_cell(d, "1h", "1d", int(d["ts"].iloc[-1]) + 3_600_000, CFG)
        self.assertEqual((c["status"], c["n_candles"], c["zones"]), ("not_enough_structure", 24, []))

    def test_gapped_cell_says_so(self):
        d = synth(1500)
        d = d[~d.index.isin(range(300, 900))].reset_index(drop=True)
        c = S.compute_cell(d, "5m", "3d", int(d["ts"].iloc[-1]) + 300_000, CFG)
        self.assertIn(c["status"], ("gapped", "short_history"))
        self.assertEqual(c["zones"], [])

    def test_cluster_width(self):
        cands = [dict(price=p, kind="pivot_high", ts=0, scan_from=0, retro=False) for p in (100.0, 100.1, 105.0)]
        z = C.cluster(cands, 0.3)
        self.assertEqual(len(z), 2)
        self.assertEqual(z[0]["n_candidates"], 2)

    def test_min_touches_filters(self):
        d = synth()
        as_of = int(d["ts"].iloc[-1]) + 300_000
        a = S.compute_cell(d, "5m", "1w", as_of, {**CFG, "min_touches": 0})
        b = S.compute_cell(d, "5m", "1w", as_of, {**CFG, "min_touches": 3})
        self.assertLessEqual(len(b["zones"]), len(a["zones"]))
        self.assertTrue(all(z["n_touches"] >= 3 for z in b["zones"]))


try:
    import fastapi  # noqa: F401
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi not installed")
class ToggleOff(unittest.TestCase):
    def test_off_computes_and_reads_nothing(self):
        import asyncio
        from app.levels import routes
        LC._cfg = dict(LC.DEFAULTS)
        old = S.levels_for
        S.levels_for = lambda *a, **k: (_ for _ in ()).throw(AssertionError("levels computed while toggle is off"))
        try:
            self.assertEqual(asyncio.run(routes.levels("binance", "BTC/USDT")), {"enabled": False})
        finally:
            S.levels_for, LC._cfg = old, None


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- window stats + confluence
from app.levels import confluence as CF, windows as WN

W15 = 900_000


def bars15(rows, start=T0):
    return pd.DataFrame([dict(ts=start + i * W15, open=o, high=h, low=l, close=c, volume=1.0) for i, (o, h, l, c) in enumerate(rows)])


class WindowStats(unittest.TestCase):
    ROWS = [(100, 100.5, 99.5, 101),   # inside zone, settles UP
            (100, 100.4, 98, 99),      # inside zone, settles DOWN
            (100, 100, 100, 100),      # inside zone, exact tie -> UP
            (90, 91, 89, 90.5),        # nowhere near the zone
            (90, 91, 89, 89.5)]
    ZONE = dict(low=99.0, high=101.0, scan_from=T0)

    def test_counts_and_tie_rule(self):
        as_of = T0 + 5 * W15
        starts, bars = WN.window_grid(bars15(self.ROWS), as_of, 5)
        s = WN.zone_stats(self.ZONE, starts, bars, [], want_table=True)
        self.assertEqual((s["windows_touched"], s["touched_settled_up"], s["touched_settled_down"]), (3, 2, 1))
        self.assertEqual((s["untouched_settled_up"], s["untouched_settled_down"]), (1, 1))
        self.assertEqual([r["settled"] for r in s["table"]], ["UP", "DOWN", "UP", "UP", "DOWN"])

    def test_n_changes_the_stats(self):
        as_of = T0 + 5 * W15
        a = WN.zone_stats(self.ZONE, *WN.window_grid(bars15(self.ROWS), as_of, 5), [])
        b = WN.zone_stats(self.ZONE, *WN.window_grid(bars15(self.ROWS), as_of, 2), [])
        self.assertNotEqual(a["windows_touched"], b["windows_touched"])

    def test_missing_windows_are_not_guessed_and_open_window_is_excluded(self):
        as_of = T0 + 5 * W15 + 400_000                      # 6th window still open
        rows = self.ROWS + [(100, 101, 99, 100.5)]
        rows[1] = (np.nan,) * 4
        starts, bars = WN.window_grid(bars15(rows), as_of, 5)
        s = WN.zone_stats(self.ZONE, starts, bars, [])
        self.assertEqual(s["windows_covered"], 4)
        self.assertEqual(int(starts[-1]), T0 + 4 * W15)     # last complete window, not the open one

    def test_touch_outcomes_are_attributed_to_their_window(self):
        touches = [dict(ts=T0 + 100, outcome="held", excursion_atr=0.8), dict(ts=T0 + W15 + 5, outcome="broke", excursion_atr=None)]
        s = WN.zone_stats(self.ZONE, *WN.window_grid(bars15(self.ROWS), T0 + 5 * W15, 5), touches)
        self.assertEqual((s["held"], s["broke"], s["avg_excursion_atr"]), (1, 1, 0.8))

    def test_service_attaches_window_stats_without_lookahead(self):
        d = synth()
        d15 = synth(3000 // 3, 900_000, seed=5)
        as_of = int(d["ts"].iloc[2200]) + 300_000
        full = S.compute_cell(d, "5m", "1w", as_of, CFG, True, d15, 24)
        self.assertGreater(len(full["zones"]), 0)
        self.assertIn("window_stats", full["zones"][0])
        poisoned = d15.copy()
        poisoned.loc[poisoned["ts"] + W15 > as_of, ["open", "high", "low", "close"]] *= 3.0
        self.assertEqual(json.dumps(full, sort_keys=True), json.dumps(S.compute_cell(d, "5m", "1w", as_of, CFG, True, poisoned, 24), sort_keys=True))


def fake_cell(name, zones, close=100.0):
    return dict(status="ok", close=close, zones=[dict(id=f"{name}:{lo}", cell=name, low=lo, high=hi, n_touches=n, side="x") for lo, hi, n in zones])


class Confluence(unittest.TestCase):
    def test_overlapping_edges_merge_across_cells(self):
        cells = [fake_cell("1h/3mo", [(101.0, 102.0, 5)]), fake_cell("15m/2w", [(101.8, 102.4, 9), (95.0, 95.5, 3)]), fake_cell("5m/3d", [(101.5, 101.9, 2)])]
        m = {g["confluence_count"]: g for g in CF.merge(cells)}
        self.assertEqual(sorted(m), [1, 3])
        g = m[3]
        self.assertEqual((g["low"], g["high"]), (101.0, 102.4))
        self.assertEqual(g["tightest"], dict(low=101.5, high=101.9))
        self.assertEqual(g["cells"], ["15m/2w", "1h/3mo", "5m/3d"])
        self.assertEqual(g["score"], 3.0)

    def test_one_cell_counts_once(self):
        m = CF.merge([fake_cell("15m/2w", [(101.0, 102.0, 4), (101.5, 102.5, 4)])])
        self.assertEqual((len(m), m[0]["confluence_count"], m[0]["score"]), (1, 1, 1.0))

    def test_weight_cap(self):
        m = CF.merge([fake_cell("a", [(1.0, 2.0, 1)]), fake_cell("b", [(1.5, 2.5, 1)])], weights={"a": 9.0, "b": 0.5})
        self.assertEqual(m[0]["score"], 1.5)

    def test_confluence_only_filters_cells(self):
        d = synth()
        as_of = int(d["ts"].iloc[-1]) + 300_000
        loader = lambda ex, sym, tf: d if tf == "5m" else synth(1000, 900_000, seed=5)
        cfg = {**CFG, "spans": ["1w"], "tfs": ["5m"], "confluence_only": True}
        r = S.levels_for("x", "BTC", cfg, ["5m", "15m"], ["1w", "3d"], as_of, loader, 24)
        self.assertTrue(all(g["confluence_count"] >= 2 for g in r["confluence"]))
        keep = {i for g in r["confluence"] for i in g["zone_ids"]}
        self.assertTrue(all(z["id"] in keep for c in r["cells"] if c["status"] == "ok" for z in c["zones"]))


# ---------------------------------------------------------------- forward journal
from app.levels import forward_log as FL

WS = T0 + 2100 * 300_000            # a quarter-hour boundary (T0 is on the hour)
assert WS % W15 == 0


def journal_world():
    d5 = synth()
    d15 = synth(1000, 900_000, seed=5)
    c0 = float(d5[d5["ts"] < WS]["close"].iloc[-1])
    rows = [(c0 + 0.02 * i, c0 + 0.02 * i + 0.3, c0 + 0.02 * i - 0.3, c0 + 0.02 * i + 0.01) for i in range(15)]
    d1 = pd.DataFrame([dict(ts=WS + i * 60_000, open=o, high=h, low=l, close=c, volume=1.0) for i, (o, h, l, c) in enumerate(rows)])
    loader = lambda ex, sym, tf: {"5m": d5, "15m": d15, "1m": d1}[tf]
    return d1, loader


JCFG = dict(CFG, tfs=["5m"], spans=["1w"], windows=24, journal=True, min_touches=1)


class Journal(unittest.TestCase):
    def setUp(self):
        self.conn = FL.connect(":memory:")
        self.d1, self.loader = journal_world()

    def test_snapshot_records_levels_and_settings_once(self):
        self.assertEqual(FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 5_000, self.loader), "written")
        self.assertEqual(FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 6_000, self.loader), "exists")
        w = FL.recent(self.conn, "x", "BTC/USDT")[0]
        self.assertEqual((w["status"], w["settings"]["windows"], w["as_of"]), ("open", 24, WS))
        self.assertGreater(len(w["zones"]), 0)
        self.assertEqual(w["n_zones"], len(w["zones"]))

    def test_late_snapshot_is_skipped_not_backfilled(self):
        self.assertEqual(FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + FL.LATE_MS + 1, self.loader), "late")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM levels_journal_windows").fetchone()[0], 0)

    def test_snapshot_ignores_the_future(self):
        d5 = synth()
        poisoned = d5.copy()
        poisoned.loc[poisoned["ts"] >= WS, ["open", "high", "low", "close"]] *= 3.0
        c2 = FL.connect(":memory:")
        FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 5_000, self.loader)
        FL.snapshot(c2, "x", "BTC/USDT", WS, JCFG, WS + 5_000, lambda ex, s, tf: poisoned if tf == "5m" else self.loader(ex, s, tf))
        pick = lambda c: [(z["zone_id"], z["low"], z["high"], z["n_touches"]) for z in c.execute("SELECT * FROM levels_journal_zones ORDER BY zone_id")]
        self.assertEqual(pick(self.conn), pick(c2))

    def test_settles_only_after_the_window_closed_and_records_touches(self):
        FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 5_000, self.loader)
        self.assertEqual(FL.settle_pending(self.conn, WS + W15 - 1, self.loader), 0)
        self.assertEqual(FL.settle_pending(self.conn, WS + W15 + FL.SETTLE_DELAY_MS, self.loader), 1)
        w = FL.recent(self.conn, "x", "BTC/USDT")[0]
        self.assertEqual((w["status"], w["settled"]), ("settled", "UP"))     # last close is above the first open in this fixture
        for z in w["zones"]:
            overlap = bool(((self.d1["low"] <= z["high"]) & (self.d1["high"] >= z["low"])).any())
            self.assertEqual(bool(z["touched"]), overlap)
        self.assertTrue(all(z["close_side"] in ("above", "below", "inside") for z in w["zones"]))

    def test_tie_settles_up(self):
        d1 = self.d1.copy()
        d1["close"] = d1["close"].astype(float)
        d1.loc[d1.index[-1], "close"] = d1["open"].iloc[0]
        FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 5_000, self.loader)
        FL.settle_pending(self.conn, WS + W15 + 60_000, lambda ex, s, tf: d1)
        self.assertEqual(FL.recent(self.conn, "x", "BTC/USDT")[0]["settled"], "UP")

    def test_missing_candles_end_as_no_data_not_a_guess(self):
        FL.snapshot(self.conn, "x", "BTC/USDT", WS, JCFG, WS + 5_000, self.loader)
        sparse = self.d1.iloc[:5]
        self.assertEqual(FL.settle_pending(self.conn, WS + W15 + 60_000, lambda *a: sparse), 0)
        self.assertEqual(FL.recent(self.conn, "x", "BTC/USDT")[0]["status"], "open")           # keep waiting
        FL.settle_pending(self.conn, WS + W15 + FL.GIVE_UP_MS + 1, lambda *a: sparse)
        self.assertEqual(FL.recent(self.conn, "x", "BTC/USDT")[0]["status"], "no_data")

    def test_toggle_off_or_journal_off_does_nothing(self):
        boom = lambda *a: (_ for _ in ()).throw(AssertionError("computed while off"))
        for cfg in (dict(JCFG, enabled=False), dict(JCFG, journal=False)):
            self.assertEqual(FL.tick(WS + 5_000, self.conn, cfg, boom, [("x", "BTC/USDT")]), {})
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM levels_journal_windows").fetchone()[0], 0)

    def test_tick_writes_then_summary_is_honest(self):
        r = FL.tick(WS + 5_000, self.conn, JCFG, self.loader, [("x", "BTC/USDT")])
        self.assertEqual(r["x:BTC/USDT"], "written")
        FL.tick(WS + W15 + 30_000, self.conn, JCFG, self.loader, [])
        s = FL.summary(self.conn, "x", "BTC/USDT")
        self.assertEqual(s["windows_settled"], 1)
        self.assertIn("Descriptive only", s["note"])
        self.assertEqual(s["any_zone_touched"]["windows"] + s["no_zone_touched"]["windows"], 1)

    def test_wilson_interval(self):
        lo, hi = FL.wilson(5, 10)
        self.assertTrue(lo < 0.5 < hi and hi - lo > 0.3)
        self.assertIsNone(FL.wilson(0, 0))


# ---------------------------------------------------------------- ML layer
from app.levels import ml as ML, scoring as SC, training as TR


def barrier_world(n, tf_ms, seed=11):
    """Price reflected between hard walls at 95 and 105, so real support/resistance genuinely holds."""
    r = np.random.default_rng(seed)
    p, out = 100.0, []
    for _ in range(n):
        o = p
        p = o + r.normal(0, 0.35)
        p = 210 - p if p > 105 else 190 - p if p < 95 else p
        out.append((o, max(o, p) + abs(r.normal(0, 0.05)), min(o, p) - abs(r.normal(0, 0.05)), p))
    o, h, l, c = zip(*out)
    return pd.DataFrame(dict(ts=T0 + np.arange(n, dtype="int64") * tf_ms, open=o, high=h, low=l, close=c, volume=r.uniform(1, 10, n)))


def frames_for(maker):
    return {"5m": maker(6000, 300_000), "15m": maker(2000, 900_000)}


ML_CFG = dict(CFG, min_touches=0)
_EV = {}


def events(kind="synth"):
    if kind not in _EV:
        mk = (lambda n, ms: synth(n, ms, seed=3 if ms == 300_000 else 5)) if kind == "synth" else barrier_world
        _EV[kind] = ML.build_events(frames_for(mk), "BTC", ML_CFG, ["5m", "15m"], ["1d", "3d"], n_cp=30, step_ms=4 * 3_600_000)
    return _EV[kind]


class MlValidation(unittest.TestCase):
    def test_purged_folds_never_train_on_an_outcome_that_reaches_the_test_block(self):
        ts = np.arange(0, 4000, 5)
        end = ts + 900                                   # long outcome windows
        for _, tr, te in ML.purged_folds(ts, end, 4):
            self.assertTrue(te.any() and tr.any())
            self.assertLess(end[tr].max(), ts[te].min())
            self.assertFalse((tr & te).any())

    def test_holdout_is_never_seen_by_the_model_calibrator_or_threshold(self):
        res, bad = ML.fit_and_validate(events())
        self.assertIsNone(bad)
        bundle, dev, hold, rand, oof, have, p_hold, info = res
        self.assertLess(dev["end"].max(), info["holdout_start"])
        self.assertGreaterEqual(hold["ts"].min(), info["holdout_start"])
        self.assertEqual(len(p_hold), len(hold))
        self.assertTrue(np.all((p_hold >= 0) & (p_hold <= 1)))

    def test_events_are_out_of_sample_relative_to_their_zone_and_timed_inside_their_checkpoint(self):
        ev = events()
        real = ev[ev["is_random"] == 0]
        self.assertTrue(((real["ts"] >= real["cp"]) & (real["ts"] < real["cp"] + 4 * 3_600_000)).all())
        self.assertTrue((real["end"] > real["ts"]).all())
        self.assertTrue(set(real["y"].unique()) <= {0.0, 1.0})
        self.assertGreater(len(ev[ev["is_random"] == 1]), 0)          # random-level baseline exists

    def test_events_ignore_candles_after_their_own_outcome(self):
        fr = frames_for(lambda n, ms: synth(n, ms, seed=3 if ms == 300_000 else 5))
        x = int(fr["5m"]["ts"].iloc[5600])
        poisoned = {tf: f.copy() for tf, f in fr.items()}
        for f in poisoned.values():
            f.loc[f["ts"] >= x, ["open", "high", "low", "close"]] *= 2.0
        a = ML.build_events(fr, "BTC", ML_CFG, ["5m", "15m"], ["1d"], 20, 4 * 3_600_000)
        b = ML.build_events(poisoned, "BTC", ML_CFG, ["5m", "15m"], ["1d"], 20, 4 * 3_600_000)
        early = lambda e: e[e["end"] < x].sort_values(["cell", "zone_id", "ts"]).reset_index(drop=True)
        self.assertGreater(len(early(a)), 0)
        pd.testing.assert_frame_equal(early(a), early(b))

    def test_effective_sample_size_collapses_touches_inside_one_horizon(self):
        base = dict(symbol="s", zone_id="z", cp=0)
        df = pd.DataFrame([dict(base, ts=t, end=t + 900_000, y=1.0) for t in (0, 60_000, 120_000, 2_000_000, 2_060_000)])
        self.assertEqual(len(set(ML.clusters(df))), 2)
        self.assertEqual(ML.boot_rate(df["y"], ML.clusters(df))["n_eff"], 2)
        self.assertEqual(ML.boot_rate(df["y"][:3], ML.clusters(df)[:3])["n_eff"], 1)      # a subset must not inherit empty clusters

    def test_bootstrap_interval_honesty(self):
        r = np.random.default_rng(0)
        y = r.integers(0, 2, 600).astype(float)
        indep = ML.boot_rate(y, np.arange(600))
        self.assertTrue(indep["lo"] < indep["rate"] < indep["hi"] and indep["hi"] - indep["lo"] < 0.12)
        dup = ML.boot_rate(np.repeat(y[:60], 10), np.repeat(np.arange(60), 10))      # 600 rows, only 60 independent
        self.assertGreater(dup["hi"] - dup["lo"], (indep["hi"] - indep["lo"]) * 2)
        self.assertEqual(dup["n_eff"], 60)

    def test_edge_flag_rules(self):
        self.assertEqual(ML.edge_flag(10, dict(diff=0.4, lo=0.2, hi=0.6)), "unproven")     # too few independent observations
        self.assertEqual(ML.edge_flag(100, dict(diff=0.1, lo=0.02, hi=0.2)), "measured")
        self.assertEqual(ML.edge_flag(100, dict(diff=0.05, lo=-0.03, hi=0.13)), "noise")    # interval includes no-edge
        self.assertEqual(ML.edge_flag(100, None), "unproven")

    def test_report_is_honest_about_what_it_did_not_prove(self):
        bundle, rep = ML.train_report(events())
        self.assertEqual(rep["status"], "ok")
        self.assertIn("holdout", rep["headline_source"])
        self.assertEqual(rep["holdout"]["metrics"]["n"], rep["holdout_events"])
        self.assertTrue(all(c["edge"] in ("measured", "unproven", "noise") for c in rep["cells_holdout"].values()))
        self.assertTrue(any("random levels" in v for v in rep["verdicts"]))
        self.assertTrue(any("cells were tested" in w for w in rep["warnings"]))
        for v in rep["cells_holdout"].values():
            self.assertLessEqual(v["n_eff"], v["n"])

    def test_too_few_events_returns_no_model(self):
        bundle, rep = ML.train_report(events().head(60))
        self.assertIsNone(bundle)
        self.assertEqual(rep["status"], "insufficient_events")

    def test_real_structure_beats_random_levels_when_it_really_holds(self):
        bundle, rep = ML.train_report(events("barrier"))
        cells = rep["cells_oof"]
        self.assertTrue(any(c["edge_vs_random"] and c["edge_vs_random"]["diff"] > 0 for c in cells.values()))
        self.assertGreater(np.mean([c["edge_vs_random"]["diff"] for c in cells.values() if c["edge_vs_random"]]), 0)


class MlLive(unittest.TestCase):
    def test_train_score_and_report_roundtrip(self):
        fr = frames_for(lambda n, ms: synth(n, ms, seed=3 if ms == 300_000 else 5))
        loader = lambda ex, s, tf: fr.get(tf, pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]))
        old = SC.app_config.MODEL_DIR
        with tempfile.TemporaryDirectory() as tmp:
            SC.app_config.MODEL_DIR, LC._cfg = Path(tmp), None
            try:
                LC.update(dict(enabled=True, tfs=["5m", "15m"], spans=["1d", "3d"], min_touches=0))
                as_of = int(fr["5m"]["ts"].iloc[-1]) + 300_000
                cfg = LC.get()
                before = S.levels_for("ex", "BTC", cfg, as_of=as_of, loader=loader, windows=24)
                self.assertFalse(before["model"]["loaded"])
                self.assertTrue(all(z["edge"] == "unproven" and "reaction_prob" not in z for c in before["cells"] if c["status"] == "ok" for z in c["zones"]))
                rep = TR.run_train("ex", ["BTC"], 30, 4.0, 0.2, None, None, None, loader)
                self.assertEqual(rep["status"], "ok")
                self.assertTrue((Path(tmp) / "levels_report_ex.json").exists() and SC.bundle_path("ex").exists())
                after = S.levels_for("ex", "BTC", cfg, as_of=as_of, loader=loader, windows=24)
                self.assertTrue(after["model"]["loaded"])
                zs = [z for c in after["cells"] if c["status"] == "ok" for z in c["zones"]]
                self.assertGreater(len(zs), 0)
                for z in zs:
                    self.assertTrue(0.0 <= z["reaction_prob"] <= 1.0)
                    self.assertIn(z["edge"], ("measured", "unproven", "noise"))
                    self.assertIn("n_touches", z)
                self.assertEqual(json.dumps(before["cells"][0]["zones"][0]["id"]), json.dumps(after["cells"][0]["zones"][0]["id"]))   # scoring never changes which zones exist
                g = S.grid("ex", "BTC", cfg, as_of, loader)
                self.assertEqual(len(g["cells"]), 75)                          # 5 timeframes x 15 spans (1y is an alias of 12mo)
                self.assertTrue(all(c["status"] in ("ok", "not_enough_structure", "short_history", "gapped") for c in g["cells"]))
                self.assertTrue(any(c["status"] != "ok" for c in g["cells"]))   # thin cells are labelled, not hidden
            finally:
                SC.app_config.MODEL_DIR, LC._cfg = old, None
                SC.forget("ex")


# ---------------------------------------------------------------- backfill policy + coverage
from app.levels import backfill as BF

NOW = T0 + 800 * 86_400_000


class BackfillPolicy(unittest.TestCase):
    def test_native_timeframes_get_two_years_and_1m_is_capped(self):
        calls = []
        fake = lambda ex, s, days, tf: (calls.append((s, tf, days)) or dict(exchange=ex, symbol=s, tf=tf, fetched=1))
        BF.run("coinbase", ["BTC/USD"], ("1m", "5m", "1h"), 730, fake)
        self.assertEqual(calls, [("BTC/USD", "1m", BF.MAX_1M_DAYS), ("BTC/USD", "5m", 730), ("BTC/USD", "1h", 730)])
        self.assertEqual(BF.status()["state"], "done")
        self.assertIn("capped", BF.status()["results"][0]["note"])

    def test_one_failure_does_not_hide_the_rest(self):
        def fake(ex, s, days, tf):
            if tf == "15m":
                raise RuntimeError("rate limited")
            return dict(fetched=5)
        BF.run("coinbase", ["BTC/USD"], ("5m", "15m", "30m"), 30, fake)
        st = BF.status()
        self.assertEqual(st["state"], "done_with_errors")
        self.assertEqual([bool(r.get("error")) for r in st["results"]], [False, True, False])

    def test_concurrent_run_is_refused(self):
        BF._lock.acquire()
        try:
            with self.assertRaises(RuntimeError):
                BF.run("x", ["BTC/USD"], ("5m",), 1, lambda *a: {})
        finally:
            BF._lock.release()


class Coverage(unittest.TestCase):
    def loader(self, gap=False):
        def load(ex, sym, tf):
            ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000}[tf]
            if tf == "1m":
                span = 5 * 86_400_000                       # only 5 days of 1m
            else:
                span = 730 * 86_400_000
            ts = np.arange(NOW - span, NOW, ms, dtype="int64")
            if gap and tf == "1h":
                ts = ts[~((ts > NOW - 20 * 86_400_000) & (ts < NOW - 10 * 86_400_000))]
            return ts
        return load

    def test_cells_are_labelled_honestly(self):
        cov = BF.coverage("x", "BTC/USD", self.loader(), NOW)
        cells = {(c["tf"], c["span"]): c for c in cov["cells"]}
        self.assertEqual(len(cov["cells"]), 75)
        self.assertEqual(cells[("1h", "2y")]["status"], "ok")
        self.assertEqual(cells[("15m", "12mo")]["status"], "ok")
        self.assertEqual(cells[("1m", "2y")]["status"], "short_history")       # only 5 days of 1m are stored
        self.assertEqual(cells[("1m", "1d")]["status"], "ok")
        self.assertEqual(cells[("1h", "1d")]["status"], "not_enough_structure")  # 24 candles
        self.assertEqual(cells[("1h", "1d")]["n_candles"], 24)

    def test_gaps_are_reported_and_gapped_cells_say_so(self):
        cov = BF.coverage("x", "BTC/USD", self.loader(gap=True), NOW)
        h = next(t for t in cov["timeframes"] if t["tf"] == "1h")
        self.assertEqual(h["gaps"], 1)
        self.assertEqual(h["worst_gaps"][0]["missing_candles"], 239)             # 10 days of hourly candles
        cells = {(c["tf"], c["span"]): c for c in cov["cells"]}
        self.assertEqual(cells[("1h", "3w")]["status"], "gapped")                # 10 of 21 days missing
        self.assertEqual(cells[("1h", "1w")]["status"], "ok")                    # the gap is outside this span

    def test_status_agrees_with_the_levels_service(self):
        ts = self.loader()("x", "s", "5m")
        d = pd.DataFrame(dict(ts=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0))
        for sp in ("1d", "1w", "1mo"):
            _, _, bad = S.gate(d, "5m", sp, NOW)
            self.assertEqual((bad or {"status": "ok"})["status"], BF._cell(ts, 300_000, S.SPAN_MS[sp], NOW)["status"])

    def test_no_candles_is_stated(self):
        cov = BF.coverage("x", "BTC/USD", lambda *a: np.array([], dtype="int64"), NOW)
        self.assertTrue(all(t["status"] == "no_candles" for t in cov["timeframes"]))
        self.assertTrue(all(c["status"] == "not_enough_structure" for c in cov["cells"]))

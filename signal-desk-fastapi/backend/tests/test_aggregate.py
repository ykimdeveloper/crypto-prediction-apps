import asyncio
import os
import sys
import time
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import aggregate, db  # noqa: E402
from app.constants import MIN_MS  # noqa: E402
from app.hub import Hub  # noqa: E402
from test_core import FakeClient  # noqa: E402


def row(t, o, h, l, c, v):
    return dict(t=t, o=o, h=h, l=l, c=c, v=v)


class Merge(unittest.TestCase):
    def test_vwap_weights_by_volume_and_stays_a_valid_candle(self):
        a = {0: row(0, 100, 101, 99, 100, 3.0)}
        b = {0: row(0, 104, 105, 103, 104, 1.0)}
        cs, st = aggregate.merge_candles({"a": a, "b": b}, mode="vwap", max_dev=0)
        self.assertAlmostEqual(cs[0]["c"], 101.0)          # (100*3 + 104*1) / 4
        self.assertEqual(cs[0]["n"], 2)
        self.assertEqual(cs[0]["v"], 4.0)
        self.assertGreaterEqual(cs[0]["h"], max(cs[0]["o"], cs[0]["c"]))

    def test_median_and_zero_volume_fallback(self):
        rows = {x: {0: row(0, p, p + 1, p - 1, p, 0.0)} for x, p in (("a", 100), ("b", 102), ("c", 110))}
        cs, _ = aggregate.merge_candles(rows, mode="median", max_dev=0)
        self.assertEqual(cs[0]["c"], 102)
        cs, _ = aggregate.merge_candles(rows, mode="vwap", max_dev=0)  # all volumes 0 -> equal weights, no ZeroDivisionError
        self.assertAlmostEqual(cs[0]["c"], 104.0)

    def test_outlier_venue_is_rejected_only_with_three_or_more(self):
        rows = {"a": {0: row(0, 100, 100, 100, 100, 1)}, "b": {0: row(0, 100.1, 100.1, 100.1, 100.1, 1)},
                "bad": {0: row(0, 130, 130, 130, 130, 1)}}
        cs, st = aggregate.merge_candles(rows, mode="vwap", max_dev=0.03)
        self.assertLess(cs[0]["c"], 101)
        self.assertEqual((st["bad"]["rejected"], st["a"]["used"]), (1, 1))
        two, _ = aggregate.merge_candles({k: rows[k] for k in ("a", "bad")}, mode="vwap", max_dev=0.03)
        self.assertGreater(two[0]["c"], 110)                 # with two venues there is no way to know who is wrong

    def test_late_venue_and_gaps(self):
        a = {0: row(0, 1, 1, 1, 1, 1), 60000: row(60000, 2, 2, 2, 2, 1)}
        b = {60000: row(60000, 4, 4, 4, 4, 1)}
        cs, _ = aggregate.merge_candles({"a": a, "b": b}, mode="vwap", max_dev=0)
        self.assertEqual([(c["t"], c["n"]) for c in cs], [(0, 1), (60000, 2)])

    def test_pick_symbol(self):
        m = {"BTC/USD": {"spot": True}, "BTC/USDT": {"spot": False}, "ETH/KRW": {"spot": True}, "SOL/USDC": {"active": False}}
        self.assertEqual(aggregate.pick_symbol(m, "BTC"), "BTC/USD")   # USDT market is a swap here, so USD wins
        self.assertIsNone(aggregate.pick_symbol(m, "ETH"))              # KRW only
        self.assertIsNone(aggregate.pick_symbol(m, "SOL"))              # inactive

    def test_id_roundtrip_is_order_independent(self):
        self.assertEqual(aggregate.agg_id(["kraken", "binance"]), aggregate.agg_id(["binance", "kraken"]))
        self.assertEqual(aggregate.members(aggregate.agg_id(["kraken", "binance"])), ["binance", "kraken"])


class Shifted(FakeClient):
    def __init__(self, markets, shift):
        super().__init__()
        self.markets = markets
        for r in self.rows:
            for i in (1, 2, 3, 4):
                r[i] += shift
        self.shift = shift

    async def watch_ohlcv(self, symbol, tf):
        await asyncio.sleep(0.05)
        self.rows[-1][4] += 0.01
        return self.rows


class HubAgg(unittest.IsolatedAsyncioTestCase):
    async def test_consolidated_feed_end_to_end(self):
        db.init(":memory:")
        clients = {
            "one": Shifted({"BTC/USDT": {"spot": True}}, 0.0),
            "two": Shifted({"BTC/USD": {"spot": True}}, 0.3),
            "krw": Shifted({"BTC/KRW": {"spot": True}}, 0.0),
        }

        async def factory(ex):
            if ex == "broken":
                raise RuntimeError("boom")
            return clients[ex]

        hub = Hub(client_factory=factory)
        aid = aggregate.agg_id(["one", "two", "krw", "broken"])
        s = await hub.add_session(aid, "AGG")
        await hub.set_symbols(s, ["BTC"])
        for _ in range(60):
            await asyncio.sleep(0.1)
            hub.tick()
            st = hub.syms[(aid, "BTC/AGG")]
            if st.eng.candles and len(st.feeds) == 2 and len(st.venue_err) == 2:
                break
        hub.tick()
        msg = [m for m in list(s.queue._queue) if m["type"] == "state"][-1]
        card = msg["cards"][0]
        self.assertFalse(card["failed"])
        self.assertIsNotNone(card["ind"]["bb"]); self.assertIsNotNone(card["ind"]["stoch"]); self.assertIsNotNone(card["ind"]["atr"])
        by = {v["exchange"]: v for v in card["venues"]}
        self.assertEqual(by["one"]["status"], "live"); self.assertEqual(by["two"]["status"], "live")
        self.assertIn("BTC/USD", by["two"]["symbol"])
        self.assertIn("no BTC", by["krw"]["error"]); self.assertIn("boom", by["broken"]["error"])
        px = card["price"]
        self.assertTrue(by["one"]["price"] < px < by["two"]["price"])   # consolidated price sits between the venues
        self.assertIsNotNone(card["spotBook"])
        # stored under the agg id, so SQLite/paper never mix with a single exchange's rows
        self.assertGreater(db._conn.execute("SELECT COUNT(*) FROM candles WHERE exchange=? AND symbol='BTC/AGG'", (aid,)).fetchone()[0], 200)
        self.assertGreater(db._conn.execute("SELECT COUNT(*) FROM indicators WHERE exchange=? AND stoch_k IS NOT NULL", (aid,)).fetchone()[0], 100)
        one = hub.syms[("one", "BTC/USDT")]
        self.assertEqual(one.refs, 1)
        hub.remove_session(s)
        # nobody left: after the idle window the agg is reaped and releases its member feeds
        hub.syms[(aid, "BTC/AGG")].idle_since = 0; hub.syms[(aid, "BTC/AGG")].last_touch = 0
        hub.tick()
        self.assertNotIn((aid, "BTC/AGG"), hub.syms)
        self.assertEqual(one.refs, 0)
        await hub.stop()

    async def test_no_exchange_has_the_asset(self):
        db.init(":memory:")

        async def factory(ex):
            return Shifted({"ETH/USDT": {}}, 0)

        hub = Hub(client_factory=factory)
        out = await hub.aggregate_snapshot(["a", "b"], "BTC", wait_s=5)
        self.assertIn("no selected exchange has it", out["error"])
        await hub.stop()


if __name__ == "__main__":
    unittest.main()

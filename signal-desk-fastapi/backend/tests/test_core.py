import asyncio
import os
import sys
import time
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import config, db, indicators  # noqa: E402
from app.constants import CONTRACT_MS, MIN_MS  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.hub import Hub  # noqa: E402
from app.paper import Paper, apply, settle, summarize  # noqa: E402


def candles(n, end_ms, seed=3, start=100.0):
    rng = np.random.default_rng(seed)
    p = start + np.cumsum(rng.normal(0, 0.3, n))
    out = []
    for i in range(n):
        t = end_ms - (n - i) * MIN_MS
        o = p[i - 1] if i else start
        out.append(dict(t=t, o=o, h=max(o, p[i]) + 0.1, l=min(o, p[i]) - 0.1, c=p[i], v=1.0))
    return out


class FakeClient:
    has = {"watchOHLCV": True, "watchTicker": True}

    def __init__(self):
        self.markets = {"BTC/USDT": {}}
        self.n = 0
        now = int(time.time() * 1000)
        self.rows = [[c["t"], c["o"], c["h"], c["l"], c["c"], c["v"]] for c in candles(300, now // MIN_MS * MIN_MS)]
        cur = self.rows[-1]
        self.rows.append([cur[0] + MIN_MS, cur[4], cur[4] + .2, cur[4] - .2, cur[4] + .05, 2.0])  # forming candle

    async def fetch_ohlcv(self, symbol, tf, limit=None):
        return self.rows[-limit:] if limit else self.rows

    async def watch_ohlcv(self, symbol, tf):
        await asyncio.sleep(0.05)
        self.n += 1
        self.rows[-1][4] += 0.01
        return self.rows

    async def watch_ticker(self, symbol):
        await asyncio.sleep(0.05)
        return dict(timestamp=int(time.time() * 1000), last=self.rows[-1][4], bid=self.rows[-1][4] - .01, ask=self.rows[-1][4] + .01,
                    high=101, low=99, baseVolume=10, quoteVolume=1000, percentage=0.5)

    async def fetch_ticker(self, symbol):
        return await self.watch_ticker(symbol)

    async def close(self):
        pass


class Indicators(unittest.TestCase):
    def test_warmup_matches_js_thresholds(self):
        cs = candles(60, 60 * MIN_MS * 100)
        self.assertIsNone(indicators.latest(indicators.indicator_frame(cs[:19]))["bb"])
        self.assertIsNotNone(indicators.latest(indicators.indicator_frame(cs[:20]))["bb"])
        self.assertIsNone(indicators.latest(indicators.indicator_frame(cs[:15]))["atr"])
        self.assertIsNotNone(indicators.latest(indicators.indicator_frame(cs[:16]))["atr"])
        self.assertIsNone(indicators.latest(indicators.indicator_frame(cs[:31]))["stoch"])
        self.assertIsNotNone(indicators.latest(indicators.indicator_frame(cs[:32]))["stoch"])

    def test_frame_has_a_row_per_candle(self):
        fr = indicators.indicator_frame(candles(120, 10**12))
        self.assertEqual(len(fr), 120)
        self.assertTrue(fr["bb_mid"].iloc[19:].notna().all() and fr["bb_mid"].iloc[:19].isna().all())


class PaperTests(unittest.TestCase):
    def setUp(self):
        db.init(":memory:")

    def _eng(self, start):
        eng = Engine()
        eng.contract_start, eng.contract_strike = start, 100.0
        eng.current = dict(t=start, o=100.0, h=100.5, l=99.5, c=100.2, v=1)
        return eng

    def test_open_settle_and_stats(self):
        start = (int(time.time() * 1000) // CONTRACT_MS) * CONTRACT_MS
        eng, paper = self._eng(start), Paper()
        book = dict(pUp=90, up=dict(bid=89.5, ask=90.5), down=dict(bid=9.5, ask=10.5))
        base = dict(signal="WAIT", conf=None, reason="x")
        sig = dict(book=book, rev=dict(base), mom=dict(base), end=dict(signal="BUY UP", conf=91, ask=90.5, tp=None, reason="r"))
        opened = apply(paper, "BTC", "1m", MIN_MS, eng, sig, start + 10 * MIN_MS)
        self.assertEqual([o["stratKey"] for o in opened], ["end"])
        sig2 = dict(book=book, rev=dict(base), mom=dict(base), end=dict(signal="BUY UP", conf=91, ask=90.5, tp=None, reason="r"))
        again = apply(paper, "BTC", "1m", MIN_MS, eng, sig2, start + 11 * MIN_MS)
        self.assertEqual(again, [], "one rec per window")
        self.assertTrue(sig2["end"]["signal"].startswith("HOLD UP"))
        self.assertTrue(settle(paper, {("BTC", "1m"): eng}, start + CONTRACT_MS + 1000))
        self.assertAlmostEqual(paper.done[0]["pnl"], 100 - 90.5)  # price 100.2 > strike 100 -> UP wins
        self.assertEqual([e for e, _ in paper.journal], ["open", "close"])
        st = summarize(paper, "1m", ["BTC"], start + CONTRACT_MS + 2000)
        self.assertEqual(st["end"]["rows"][0]["all"]["n"], 1)
        self.assertEqual(st["end"]["rows"][1]["label"], "BTC")

    def test_db_roundtrip(self):
        start = 1_000_000 * CONTRACT_MS
        rec = dict(id="BTC|1m|end|1", sym="BTC", tf="1m", strat="end", dir="UP", conf=91, strike=100.0, entry=90.5, tp=None, start=start,
                   end=start + CONTRACT_MS, shadow=False, opened=start + 1, path=[90.0, 92.0])
        db.save_trade("x", "BTC/USDT", rec, "open", start)
        self.assertEqual(db.load_closed_trades("x"), [])
        db.save_trade("x", "BTC/USDT", {**rec, "pnl": 9.5, "how": "settle", "close": 101.0, "settlePnl": 9.5}, "closed", start + CONTRACT_MS)
        got = db.load_closed_trades("x")
        self.assertEqual((len(got), got[0]["sym"], got[0]["path"], got[0]["pnl"]), (1, "BTC", [90.0, 92.0], 9.5))


class HubTests(unittest.IsolatedAsyncioTestCase):
    async def test_feed_persists_and_pushes_state(self):
        db.init(":memory:")
        fake = FakeClient()

        async def factory(_):
            return fake

        hub = Hub(client_factory=factory)
        session = await hub.add_session("fake", "USDT")
        await hub.set_symbols(session, ["btc", "bad symbol!"])
        self.assertEqual(session.symbols, ["BTC"])
        st, cs = await hub.candles("fake", "BTC", "USDT", limit=50, wait_s=5)
        self.assertGreater(len(cs), 40)
        await asyncio.sleep(0.3)  # let watch_* loops turn over a few times
        hub.tick()
        msg = None
        while not session.queue.empty():
            m = session.queue.get_nowait()
            msg = m if m["type"] == "state" else msg
        self.assertEqual(msg["exchange"], "fake")
        card = msg["cards"][0]
        self.assertEqual(card["symbol"], "BTC")
        self.assertIsNotNone(card["ind"]["bb"])
        self.assertIsNotNone(card["sig"]["book"] or card["sig"]["rev"])
        self.assertEqual(set(msg["paper"]), {"rev", "mom", "end"})
        self.assertIsNotNone(card["spotBook"])

        n_c = db._conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
        n_i = db._conn.execute("SELECT COUNT(*) FROM indicators WHERE stoch_k IS NOT NULL").fetchone()[0]
        n_t = db._conn.execute("SELECT COUNT(*) FROM ticker_snapshots").fetchone()[0]
        self.assertGreater(n_c, 250)
        self.assertGreater(n_i, 200)
        self.assertEqual(n_t, 1)
        row = db.query("indicators", exchange="fake", symbol="BTC/USDT", limit=1)[0]
        self.assertIn(row["bb_state"], ("Mid_Band", "Upper_Stretched", "Lower_Stretched"))
        self.assertEqual(row["backend"], "pandas")

        hub.remove_session(session)
        self.assertEqual(st.refs, 0)
        await hub.stop()

    async def test_unlisted_pair_reports_error(self):
        db.init(":memory:")
        fake = FakeClient()

        async def factory(_):
            return fake

        hub = Hub(client_factory=factory)
        st, cs = await hub.candles("fake", "NOPE", "USDT", wait_s=3)
        self.assertEqual(cs, [])
        self.assertIn("not listed", st.error)
        await hub.stop()


if __name__ == "__main__":
    unittest.main()

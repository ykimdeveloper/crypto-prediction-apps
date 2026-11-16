import io
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

from app import config, db  # noqa: E402
from app.constants import CONTRACT_MS  # noqa: E402
from app.quant15 import cdd, csvimport, live  # noqa: E402

NOW = (1_790_000_000_000 // CONTRACT_MS) * CONTRACT_MS + 60_000


def sig(**kw):
    base = {"signal": "WAIT", "conf": None, "reason": "-"}
    out = {k: dict(base) for k in ("rev", "mom", "end")}
    for k, v in kw.items():
        out[k] = dict(signal=v[0], conf=v[1], reason="r")
    return out


class TestLivePolicy(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = config.MODEL_DIR
        config.MODEL_DIR = Path(self.tmp.name)
        live._cfg = None; live._cache.clear()

    def tearDown(self):
        config.MODEL_DIR = self.old
        live._cfg = None; live._cache.clear()
        self.tmp.cleanup()

    def wait_prior(self, ex="binance", sym="BTC/USDT"):
        for _ in range(100):
            p = live.prior(ex, sym, NOW)
            if p is not None:
                return p
            time.sleep(0.02)
        self.fail("model view never arrived")

    def test_original_is_a_noop(self):
        s = sig(end=("BUY UP", 80))
        before = dict(s["end"])
        info = live.apply("binance", "BTC/USDT", s, NOW)
        self.assertEqual(s["end"], before)
        self.assertEqual((info["mode"], info["vetoed"]), ("original", []))

    def test_gated_vetoes_disagreement_and_keeps_agreement(self):
        live.set_policy("gated", 0.02)
        with mock.patch.object(live.S, "prior_now", return_value=dict(available=True, p_up=0.44, window_start=0)), \
             mock.patch.object(live, "_bundle", return_value={}):
            self.wait_prior()
            s = sig(end=("BUY UP", 80), rev=("BUY DOWN", 80), mom=("BUY UP", 10))
            info = live.apply("binance", "BTC/USDT", s, NOW)
        self.assertEqual(s["end"]["signal"], "WAIT"); self.assertEqual(s["end"]["original"], "BUY UP")
        self.assertIn("VETO", s["end"]["reason"])
        self.assertEqual(s["rev"]["signal"], "BUY DOWN")              # 0.44 backs DOWN
        self.assertEqual(s["mom"]["signal"], "BUY UP")                # under minConf: paper.py handles it, not the policy
        self.assertEqual((info["vetoed"], info["active"], info["pUp"]), (["end"], True, 0.44))

    def test_pending_holds_buys(self):
        live.set_policy("gated")
        gate = mock.Mock(side_effect=lambda *a, **k: time.sleep(0.3) or dict(available=True, p_up=0.6))
        with mock.patch.object(live.S, "prior_now", gate), mock.patch.object(live, "_bundle", return_value={}):
            s = sig(end=("BUY UP", 80))
            info = live.apply("binance", "BTC/USDT", s, NOW)
            self.assertEqual(s["end"]["signal"], "WAIT")
            self.assertIn("waiting", s["end"]["reason"]); self.assertFalse(info["active"])
            self.wait_prior()
            s = sig(end=("BUY UP", 80))
            live.apply("binance", "BTC/USDT", s, NOW)
            self.assertEqual(s["end"]["signal"], "BUY UP")            # model backs UP at 0.60

    def test_untrained_model_passes_through_and_says_so(self):
        live.set_policy("gated")
        s = sig(end=("BUY UP", 80))
        live.apply("binance", "BTC/USDT", s, NOW)                     # schedules the lookup
        self.wait_prior()
        s = sig(end=("BUY UP", 80))
        info = live.apply("binance", "BTC/USDT", s, NOW)
        self.assertEqual(s["end"]["signal"], "BUY UP")
        self.assertFalse(info["active"]); self.assertIn("no trained model", info["note"])

    def test_policy_persists_and_rejects_junk(self):
        live.set_policy("gated", 0.05)
        live._cfg = None
        self.assertEqual(live.get(), dict(policy="gated", margin=0.05))
        with self.assertRaises(ValueError):
            live.set_policy("model_only")


class FakeResp:
    def __init__(self, body): self.body = body
    def raise_for_status(self): pass
    def iter_content(self, n): yield self.body


class FakeHttp:
    def __init__(self, body): self.body, self.calls = body, 0
    def get(self, url, **kw):
        self.calls += 1
        self.url = url
        return FakeResp(self.body)


def cdd_csv(n):
    rows = ["https://www.CryptoDataDownload.com", "Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount"]
    t0 = NOW // 60_000 * 60_000
    for i in range(n - 1, -1, -1):
        p = 100 + (i % 50) * 0.1
        rows.append(f"{(t0 - (n - 1 - i) * 60_000) * 0 + t0 - (n - 1 - i) * 60_000},x,BTC/USDT,{p},{p+1},{p-1},{p+.5},3,300,9")
    return ("\n".join(rows) + "\n").encode()


class TestCdd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = cdd.CACHE
        cdd.CACHE = Path(self.tmp.name)
        db.init(":memory:")

    def tearDown(self):
        cdd.CACHE = self.old

    def test_url_and_non_binance_refused(self):
        self.assertTrue(cdd.url_for("BTC/USDT").endswith("Binance_BTCUSDT_minute.csv"))
        with self.assertRaises(ValueError):
            cdd.fetch_into("BTC/USDT", "okx", http=FakeHttp(b""))

    def test_html_page_is_not_saved_as_data(self):
        with self.assertRaises(RuntimeError):
            cdd.download("BTC/USDT", http=FakeHttp(b"<html><body>Please log in</body></html>"))
        self.assertFalse(list(Path(self.tmp.name).glob("*.csv")))

    def test_ensure_fetches_once_then_uses_db(self):
        http = FakeHttp(cdd_csv(3 * 1440))
        r = cdd.ensure("BTC/USDT", "binance", days=3, min_windows=100, http=http)
        self.assertTrue(r["fetched"]); self.assertGreaterEqual(r["windows"], 100)
        r2 = cdd.ensure("BTC/USDT", "binance", days=3, min_windows=100, http=http)
        self.assertFalse(r2["fetched"]); self.assertEqual(http.calls, 1)

    def test_old_file_is_flagged(self):
        r = cdd.fetch_into("BTC/USDT", "binance", days=2, http=FakeHttp(cdd_csv(2 * 1440)))
        self.assertGreater(r["newest_age_days"], 3)
        self.assertIn("warning", r)

    def test_zip_of_headerless_vision_file(self):
        import zipfile
        z = Path(self.tmp.name) / "v.zip"
        body = "\n".join(f"{(NOW // 60_000 * 60_000 + i * 60_000) * 1000},1,2,0.5,1.5,9,0,0,0,0,0,0" for i in range(40))
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("BTCUSDT-1m-2026-09-01.csv", body)
        d = csvimport.read_1m(str(z), "BTC/USDT")
        self.assertEqual(len(d), 40)
        self.assertEqual(int(d["ts"].iloc[0]), NOW // 60_000 * 60_000)

    def test_newest_first_head_read_matches_full_read(self):
        p = Path(self.tmp.name) / "big.csv"; p.write_bytes(cdd_csv(4 * 1440 + 6000))
        d = csvimport.read_1m(str(p), "BTC/USDT", days=1)
        self.assertEqual(len(d), 1440)
        self.assertEqual(int(d["ts"].iloc[-1]), NOW // 60_000 * 60_000)


if __name__ == "__main__":
    unittest.main()

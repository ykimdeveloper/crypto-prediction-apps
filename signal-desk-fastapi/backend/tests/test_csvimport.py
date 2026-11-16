import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app import db  # noqa: E402
from app.quant15 import csvimport as C  # noqa: E402

T0 = 1_790_000_000_000 // 60_000 * 60_000


def write_cdd(path, n, unit_mult=1, drop=()):
    rows = ["https://www.CryptoDataDownload.com",
            "Unix,Date,Symbol,Open,High,Low,Close,Volume BTC,Volume USDT,tradecount"]
    for i in range(n - 1, -1, -1):           # newest first, like the real files
        if i in drop:
            continue
        ts = (T0 + i * 60_000) * unit_mult
        p = 100 + i * 0.01
        rows.append(f"{ts},2026-01-01 00:00:00,BTC/USDT,{p},{p+1},{p-1},{p+0.5},7,700,42")
    Path(path).write_text("\n".join(rows) + "\n")


class TestCsvImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db.init(":memory:")

    def test_cdd_banner_order_and_volume_column(self):
        p = Path(self.tmp.name) / "a.csv"; write_cdd(p, 500)
        d = C.read_1m(str(p), "BTC/USDT")
        self.assertEqual(len(d), 500)
        self.assertTrue(d["ts"].is_monotonic_increasing)
        self.assertEqual(float(d["v"].iloc[0]), 7.0)          # Volume BTC, not Volume USDT (700)

    def test_units_ms_us_s_agree(self):
        out = []
        for mult, name in ((1, "ms"), (1000, "us")):
            p = Path(self.tmp.name) / f"{name}.csv"; write_cdd(p, 50, unit_mult=mult)
            out.append(C.read_1m(str(p), "BTC/USDT")["ts"].tolist())
        self.assertEqual(out[0], out[1])
        self.assertEqual(out[0][0], T0)

    def test_days_trim_and_gaps(self):
        p = Path(self.tmp.name) / "g.csv"; write_cdd(p, 3 * 1440, drop={100, 101, 102})
        r = C.import_csv([str(p)], "binance", "BTC/USDT", days=1)
        self.assertEqual(r["rows"], 1440)
        self.assertEqual(r["missing_minutes"], 0)               # the holes are older than the 1-day cut
        r = C.import_csv([str(p)], "binance", "BTC/USDT", days=3)
        self.assertEqual(r["missing_minutes"], 3)
        self.assertEqual(r["largest_gap_min"], 3)
        self.assertEqual(len(db.query("candles", exchange="binance", symbol="BTC/USDT", tf="1m", limit=10000)), 3 * 1440 - 3)

    def test_binance_vision_headerless(self):
        p = Path(self.tmp.name) / "v.csv"
        p.write_text("\n".join(f"{(T0 + i * 60_000) * 1000},1,2,0.5,1.5,9,0,0,0,0,0,0" for i in range(30)) + "\n")
        d = C.read_1m(str(p), "BTC/USDT")
        self.assertEqual(len(d), 30)
        self.assertEqual(int(d["ts"].iloc[0]), T0)


if __name__ == "__main__":
    unittest.main()

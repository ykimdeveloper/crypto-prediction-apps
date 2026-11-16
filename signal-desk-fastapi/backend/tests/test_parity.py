"""Python port vs the browser code. Needs `node`; skipped otherwise. Uses INDICATOR_BACKEND=pandas (bit-exact port)."""
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine import Engine  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def close(a, b, tol=1e-6):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def cmp(a, b, path, errs):
    if isinstance(a, dict):
        for k in a:
            cmp(a[k], b.get(k), f"{path}.{k}", errs)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        if not close(float(a), None if b is None else float(b)):
            errs.append(f"{path}: js={a} py={b}")
    elif a != b:
        errs.append(f"{path}: js={a!r} py={b!r}")


@unittest.skipUnless(shutil.which("node"), "node not installed")
class Parity(unittest.TestCase):
    def test_indicators_and_signals_match_js(self):
        out = subprocess.run(["node", "--import", "./tests/register.mjs", "backend/tests/parity_js.mjs"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        data = json.loads(out)
        candles = data["candles"]
        errs, stories = [], set()
        for case in data["cases"]:
            n, now = case["n"], case["now"]
            closed = [dict(c) for c in candles[:n]]
            forming = dict(candles[n - 1]); forming.update(t=candles[n - 1]["t"] + 60000, o=candles[n - 1]["c"],
                                                             c=candles[n - 1]["c"] + 0.03, h=candles[n - 1]["c"] + 0.1, l=candles[n - 1]["c"] - 0.1)
            eng = Engine(60000)
            eng.update(closed, forming, now)
            sig = eng.evaluate(now)
            tag = f"n={n},now={now}"
            cmp(case["ind"], eng.ind, f"{tag}.ind", errs)
            cmp({"strike": case["strike"], "contractStart": case["contractStart"]},
                {"strike": eng.contract_strike, "contractStart": eng.contract_start}, f"{tag}", errs)
            js = case["sig"]
            cmp({k: js[k] for k in ("module", "story", "lean", "rev", "mom", "end", "book")}, sig, f"{tag}.sig", errs)
            stories.add(js["story"])
        self.assertEqual(errs, [], "\n" + "\n".join(errs[:25]))
        self.assertGreaterEqual(len(stories - {None}), 2, f"test data only exercised {stories}")


if __name__ == "__main__":
    unittest.main()

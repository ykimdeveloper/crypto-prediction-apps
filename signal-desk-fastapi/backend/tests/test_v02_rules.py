"""v0.2 video rules on the server: momentum-only mid-market, 6:00 minimum, price already slightly on the lean side of the
strike (not exhausted), a high chance of the +20¢, and an end-of-market indicator nudge that fades with time.
Mirrors tests/t8_v02_rules.mjs."""
import math
import os
import sys
import unittest
from pathlib import Path

os.environ["INDICATOR_BACKEND"] = "pandas"
os.environ["ML_ENABLED"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.constants import EOM_CONF_SIGMA, MID  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.strategy import js_round, normal_cdf  # noqa: E402

NOW = 1_000_000 * 900_000 + 7 * 60_000


def sig(rem, z, k, traj, sstate="Mixed", atr=0.5, astate="Active", bbw=2):
    """z = signed distance from the strike in sigmas of the remaining path (ATR * sqrt(candles left))."""
    eng = Engine(60000)
    eng.contract_start = NOW - int((15 - rem) * 60000)
    eng.contract_strike = 100.0
    eng.current = dict(t=eng.contract_start, o=100, h=100, l=100, c=100 + z * atr * math.sqrt(rem), v=1)
    eng.last_tick_at = NOW
    eng.ind = dict(bb=dict(mid=100, upper=100 + bbw, lower=100 - bbw, state="Mid_Band"),
                   stoch=dict(k=k, traj=traj, state=sstate), atr=dict(atr=atr, state=astate))
    return eng.evaluate(NOW)


MOM_UP = dict(rem=10, z=0.4, k=70, traj="Rising")


class MidMarketRules(unittest.TestCase):
    def tearDown(self):
        MID.update(revEnabled=False, minHitProb=0.72)

    def test_rev_is_classified_but_never_buys(self):
        rev_setup = dict(rem=10, z=-0.506, k=10, traj="Rising", sstate="Bounce_Bottom", bbw=1)
        s = sig(**rev_setup)
        self.assertEqual(s["story"], "Mean Reversion")
        self.assertEqual(s["rev"]["signal"], "WAIT")
        self.assertIn("off in v0.2", s["rev"]["reason"])
        MID["revEnabled"] = True
        self.assertEqual(sig(**rev_setup)["rev"]["signal"], "BUY UP")

    def test_momentum_buys_and_reports_hit_chance(self):
        s = sig(**MOM_UP)
        self.assertEqual(s["mom"]["signal"], "BUY UP")
        self.assertGreaterEqual(s["mom"]["pHit"], MID["minHitProb"])

    def test_needs_six_minutes(self):
        self.assertEqual(sig(**{**MOM_UP, "rem": 6.0})["mom"]["signal"], "BUY UP")
        s = sig(**{**MOM_UP, "rem": 5.9})
        self.assertEqual(s["mom"]["signal"], "WAIT")
        self.assertIn("Under 6:00 left", s["mom"]["reason"])

    def test_price_must_already_be_a_little_on_the_side_but_not_exhausted(self):
        for z in (-0.3, 0.0, 0.05):
            s = sig(**{**MOM_UP, "z": z})
            self.assertEqual(s["mom"]["signal"], "WAIT")
            self.assertIn("isn't on the UP side", s["mom"]["reason"])
        self.assertIn("already ran", sig(**{**MOM_UP, "z": 1.3})["mom"]["reason"])
        self.assertEqual(sig(rem=10, z=-0.4, k=30, traj="Falling")["mom"]["signal"], "BUY DOWN")

    def test_hit_probability_bar(self):
        MID["minHitProb"] = sig(**MOM_UP)["mom"]["pHit"] + 0.02
        s = sig(**MOM_UP)
        self.assertEqual(s["mom"]["signal"], "WAIT")
        self.assertIn("% likely", s["mom"]["reason"])
        MID["minHitProb"] = 0.95
        for z in (0.2, 0.4, 0.7, 0.9):
            self.assertEqual(sig(**{**MOM_UP, "z": z})["mom"]["signal"], "WAIT")

    def test_asleep_atr_still_blocks(self):
        self.assertIn("asleep", sig(**{**MOM_UP, "astate": "Quiet/Asleep"})["mom"]["reason"])


class EndOfMarketFade(unittest.TestCase):
    @staticmethod
    def disagree(rem):  # price above the strike (END favors UP) while the indicators lean DOWN
        return sig(rem=rem, z=0.5, k=92, traj="Falling", sstate="Rollover_Top")["end"]["conf"]

    def test_confidence_recovers_as_the_window_closes(self):
        self.assertGreater(self.disagree(2), self.disagree(5))
        self.assertGreater(self.disagree(5), self.disagree(8))

    def test_no_nudge_left_at_two_minutes_and_full_nudge_at_eight(self):
        stat = normal_cdf(0.5 / EOM_CONF_SIGMA) * 100
        self.assertEqual(self.disagree(2), js_round(stat))
        self.assertLess(self.disagree(8), js_round(stat))


if __name__ == "__main__":
    unittest.main()

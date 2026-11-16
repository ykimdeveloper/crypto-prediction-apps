import unittest

from app import settlement as S


class SettlementRule(unittest.TestCase):
    def test_tie_settles_up(self):
        self.assertTrue(S.settles_up(100.0, 100.0))
        self.assertEqual(S.settle_dir(100.0, 100.0), "UP")

    def test_strict_moves(self):
        self.assertEqual(S.settle_dir(100.01, 100.0), "UP")
        self.assertEqual(S.settle_dir(99.99, 100.0), "DOWN")

    def test_positions(self):
        self.assertTrue(S.position_wins("UP", 100.0, 100.0))
        self.assertFalse(S.position_wins("DOWN", 100.0, 100.0))   # a tie pays UP only
        self.assertTrue(S.position_wins("DOWN", 99.0, 100.0))
        self.assertFalse(S.position_wins("UP", 99.0, 100.0))

    def test_exactly_one_side_wins(self):
        for end in (99.0, 100.0, 101.0):
            self.assertNotEqual(S.position_wins("UP", end, 100.0), S.position_wins("DOWN", end, 100.0))


if __name__ == "__main__":
    unittest.main()

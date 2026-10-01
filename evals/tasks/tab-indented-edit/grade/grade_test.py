import unittest
from legacy import discount


class Grade(unittest.TestCase):
    def test_behaviour(self):
        self.assertAlmostEqual(discount(60), 48.0)
        self.assertEqual(discount(50), 50)
        self.assertEqual(discount(40), 40)

    def test_still_tabs(self):
        for line in open("legacy.py"):
            self.assertFalse(line.startswith(" "), f"space-indented line: {line!r}")

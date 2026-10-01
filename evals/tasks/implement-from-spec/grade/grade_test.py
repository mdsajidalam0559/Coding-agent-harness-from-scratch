import unittest
from durations import parse_duration


class Grade(unittest.TestCase):
    def test_valid(self):
        cases = {"45s": 45, "2h": 7200, "90m": 5400, "1h30m": 5400, "1h 5m 10s": 3910, "0s": 0, "10m5s": 605}
        for text, seconds in cases.items():
            self.assertEqual(parse_duration(text), seconds, text)

    def test_invalid(self):
        for text in ["", "10", "10x", "h", "-5m", "5m1h", "1h1h", "1.5h", "abc"]:
            with self.assertRaises(ValueError, msg=text):
                parse_duration(text)

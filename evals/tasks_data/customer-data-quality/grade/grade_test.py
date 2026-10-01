import csv
import re
import unittest


def answer():
    return open("reports/answer.md").read()


def numbers(text):
    return [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def has_number(text, value, tolerance=0.01):
    return any(abs(n - value) <= tolerance for n in numbers(text))


class Grade(unittest.TestCase):
    def test_answer(self):
        rows = list(csv.DictReader(open("customers.csv")))
        ids = {r["id"] for r in rows}
        missing = {r["id"] for r in rows if not r["email"]}
        text = answer()
        self.assertTrue(has_number(text, len(ids), 0), f"expected {len(ids)} unique customers in: {text}")
        self.assertTrue(has_number(text, len(missing), 0), f"expected {len(missing)} without email in: {text}")
        self.assertFalse(has_number(text, len(rows), 0) and len(rows) != len(ids),
                         "reports the row count (with duplicates) instead of unique customers")

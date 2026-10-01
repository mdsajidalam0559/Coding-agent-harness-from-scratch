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
        revenue = {}
        for row in csv.DictReader(open("data/sales.csv")):
            if row["region"] and row["units"]:
                revenue[row["region"]] = revenue.get(row["region"], 0) + int(row["units"]) * float(row["price"])
        best = max(revenue, key=revenue.get)
        text = answer()
        self.assertIn(best, text)
        self.assertTrue(has_number(text, revenue[best]), f"expected {revenue[best]} in: {text}")

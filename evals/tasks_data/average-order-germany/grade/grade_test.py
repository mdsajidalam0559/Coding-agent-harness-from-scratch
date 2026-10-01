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
        country = {r["id"]: r["country"] for r in csv.DictReader(open("customers.csv"))}
        amounts = [float(r["amount"]) for r in csv.DictReader(open("orders.csv")) if country[r["customer_id"]] == "Germany"]
        expected = round(sum(amounts) / len(amounts), 2)
        text = answer()
        self.assertTrue(has_number(text, expected, 0.011), f"expected {expected} in: {text}")

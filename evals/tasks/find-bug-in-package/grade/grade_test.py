import unittest
from inventory.catalog import find
from inventory.models import Order
from inventory.orders import order_total


class Grade(unittest.TestCase):
    def order(self, code=None):
        return Order(lines=[(find("C3"), 2)], discount_code=code)

    def test_codes_apply(self):
        self.assertAlmostEqual(order_total(self.order("save10")), 77.04)
        self.assertAlmostEqual(order_total(self.order("SAVE10")), 77.04)
        self.assertAlmostEqual(order_total(self.order(" Half ")), 47.8)  # 40 + 7% tax + $5 shipping (under $50)

    def test_without_code_unchanged(self):
        self.assertAlmostEqual(order_total(self.order()), 85.6)
        self.assertAlmostEqual(order_total(self.order("bogus")), 85.6)

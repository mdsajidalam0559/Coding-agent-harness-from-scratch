from shop.pricing import calc_total, calc_tax


class Cart:
    def __init__(self):
        self.items = []

    def add(self, price, qty=1):
        self.items.append((price, qty))

    def checkout(self):
        subtotal = calc_total(self.items)
        return subtotal + calc_tax(subtotal)

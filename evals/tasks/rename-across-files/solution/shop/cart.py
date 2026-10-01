from shop.pricing import compute_subtotal, calc_tax


class Cart:
    def __init__(self):
        self.items = []

    def add(self, price, qty=1):
        self.items.append((price, qty))

    def checkout(self):
        subtotal = compute_subtotal(self.items)
        return subtotal + calc_tax(subtotal)

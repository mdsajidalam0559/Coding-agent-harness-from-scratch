def calc_total(items):
    total = 0
    for price, qty in items:
        total += price * qty
    return total


def calc_tax(amount):
    return amount * 0.1

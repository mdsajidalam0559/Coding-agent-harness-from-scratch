import sys

from inventory.catalog import find
from inventory.models import Order
from inventory.orders import order_total


def main(argv):
    order = Order()
    for arg in argv:
        if arg.startswith("--code="):
            order.discount_code = arg.split("=", 1)[1]
        else:
            sku, qty = arg.split(":")
            order.lines.append((find(sku), int(qty)))
    print(order_total(order))


if __name__ == "__main__":
    main(sys.argv[1:])

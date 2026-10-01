from inventory.discounts import discount_rate
from inventory.shipping import shipping_cost
from inventory.tax import tax_for
from inventory.utils import money


def subtotal(order):
    return sum(product.price * qty for product, qty in order.lines)


def order_total(order):
    base = subtotal(order)
    discounted = base * (1 - discount_rate(order.discount_code))
    return money(discounted + tax_for(discounted, order.country) + shipping_cost(discounted, order.country))

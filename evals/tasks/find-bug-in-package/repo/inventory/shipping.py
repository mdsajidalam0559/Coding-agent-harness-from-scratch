def shipping_cost(subtotal, country):
    if subtotal >= 50:
        return 0.0
    return 5.0 if country == "US" else 12.0

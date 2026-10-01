DISCOUNT_CODES = {"save10": 0.10, "half": 0.50}


def discount_rate(code):
    """Fraction to take off the subtotal for a discount code (codes are case-insensitive)."""
    if not code:
        return 0.0
    return DISCOUNT_CODES.get(code.strip().lower(), 0.0)

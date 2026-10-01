RATES = {"US": 0.07, "DE": 0.19, "IN": 0.18}


def tax_for(amount, country):
    return round(amount * RATES.get(country, 0.0), 2)

from inventory.models import Product

CATALOG = {
    "A1": Product("A1", "Notebook", 4.0),
    "B2": Product("B2", "Pen", 1.5),
    "C3": Product("C3", "Backpack", 40.0),
}


def find(sku):
    try:
        return CATALOG[sku]
    except KeyError:
        raise ValueError(f"unknown sku {sku}") from None

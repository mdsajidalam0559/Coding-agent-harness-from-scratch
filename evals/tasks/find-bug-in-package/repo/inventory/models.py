from dataclasses import dataclass, field


@dataclass
class Product:
    sku: str
    name: str
    price: float


@dataclass
class Order:
    lines: list = field(default_factory=list)  # (Product, quantity)
    discount_code: str | None = None
    country: str = "US"

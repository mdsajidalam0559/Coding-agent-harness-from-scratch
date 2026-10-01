from shop.cart import Cart

cart = Cart()
cart.add(10, 2)
cart.add(5)
print(cart.checkout())

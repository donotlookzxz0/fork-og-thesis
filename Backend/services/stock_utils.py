from models.item import Item
from db import db


def validate_cart_stock(cart, lock=False):
    """
    Validate that all items in cart exist and have enough stock.
    If lock=True, rows are locked (FOR UPDATE) for safe deduction.
    
    Returns: list of (Item, qty)
    Raises: Exception on any invalid stock
    """

    if not cart:
        raise Exception("Cart is empty")

    validated = []

    for entry in cart:
        barcode = entry.get("barcode")
        qty = entry.get("quantity")

        if not barcode or not qty or qty <= 0:
            raise Exception("Invalid cart item data")

        query = Item.query.filter_by(barcode=barcode)

        # Lock row if requested (admin approval / final deduction)
        if lock:
            query = query.with_for_update()

        item = query.first()

        if not item:
            raise Exception(f"Item not found: {barcode}")

        if item.quantity <= 0:
            raise Exception(f"Item out of stock: {item.name}")

        if qty > item.quantity:
            raise Exception(f"Insufficient stock for {item.name}")

        validated.append((item, qty))

    return validated

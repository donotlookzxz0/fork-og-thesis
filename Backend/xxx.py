# seed_items.py
from app import app, db
from models.item import Item

items_data = [
    {"name": "BEEF PREMIUM", "category": "Meat", "price": 255, "barcode": "73841926"},
    {"name": "PORK BELLY", "category": "Meat", "price": 200, "barcode": "56291748"},
    {"name": "PORK REGULAR", "category": "Meat", "price": 160, "barcode": "98417325"},
    {"name": "KIMCHI -RECTANGULAR", "category": "Vegetables", "price": 100, "barcode": "67129584"},
    {"name": "KIMCHI-CIRCULAR", "category": "Vegetables", "price": 70, "barcode": "82947163"},
    {"name": "LETTUCE 40 grams", "category": "Vegetables", "price": 40, "barcode": "59472618"},
    {"name": "MELTED CHEESE", "category": "Vegetables", "price": 95, "barcode": "41583697"},
    {"name": "FISH CAKE 240 grams", "category": "Vegetables", "price": 85, "barcode": "29768451"},
    {"name": "SHABU SHABU", "category": "Vegetables", "price": 100, "barcode": "15973824"},
    {"name": "GLASS NOODLES", "category": "Dry Goods", "price": 240, "barcode": "68734192"},
    {"name": "SESAME SEEDS", "category": "Spices & Seasonings", "price": 200, "barcode": "92385671"}
]

with app.app_context():
    for item_data in items_data:
        existing = Item.query.filter_by(barcode=item_data["barcode"]).first()
        if existing:
            print(f"Skipping existing item: {existing.name}")
            continue

        item = Item(
            name=item_data["name"],
            category=item_data["category"],
            price=item_data["price"],
            barcode=item_data["barcode"],
            quantity=200  # default starting quantity
        )
        db.session.add(item)

    db.session.commit()
    print("✅ Items seeded successfully!")

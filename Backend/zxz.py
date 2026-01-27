# seed_sales.py
from app import app, db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from datetime import datetime, timedelta
import random

USER_ID = 1

# Dates from Jan 22 to Jan 27, 2026
start_date = datetime(2026, 1, 22)
end_date = datetime(2026, 1, 27)

with app.app_context():
    items = Item.query.all()
    if not items:
        raise Exception("No items found. Run seed_items.py first.")

    current_date = start_date
    while current_date <= end_date:
        # Create 1–3 transactions per day
        for _ in range(random.randint(1, 3)):
            transaction = SalesTransaction(user_id=USER_ID, date=current_date)
            db.session.add(transaction)
            db.session.flush()  # get transaction id

            # Randomly pick 1–5 items per transaction
            chosen_items = random.sample(items, random.randint(1, min(5, len(items))))
            for item in chosen_items:
                qty = random.randint(10, 30)
                if item.quantity < qty:
                    qty = item.quantity  # avoid negative stock

                item.quantity -= qty

                db.session.add(SalesTransactionItem(
                    transaction=transaction,
                    item=item,
                    quantity=qty,
                    price_at_sale=item.price
                ))

            db.session.commit()

        current_date += timedelta(days=1)

    print("✅ Sales transactions seeded successfully!")

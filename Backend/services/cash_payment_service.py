from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.pending_cash_payment import PendingCashPayment
from models.item import Item


class CashPaymentService:

    @staticmethod
    def create_pending_payment(user_id, cart):
        """
        Create a pending cash payment request.
        Reject immediately if stock is invalid.
        """

        # 🔒 Validate stock BEFORE creating pending payment
        if not cart:
            raise Exception("Cart is empty")

        for entry in cart:
            barcode = entry.get("barcode")
            qty = entry.get("quantity")

            if not barcode or not qty or qty <= 0:
                raise Exception("Invalid cart item data")

            item = Item.query.filter_by(barcode=barcode).first()
            if not item:
                raise Exception(f"Item not found: {barcode}")

            if item.quantity <= 0:
                raise Exception(f"Item out of stock: {item.name}")

            if qty > item.quantity:
                raise Exception(f"Insufficient stock for {item.name}")

        # If user already has a pending request, return it
        existing = PendingCashPayment.query.filter_by(
            user_id=user_id,
            status="PENDING"
        ).first()

        if existing:
            return existing

        pending = PendingCashPayment(
            user_id=user_id,
            cart=cart
        )

        db.session.add(pending)
        db.session.commit()
        return pending

    @staticmethod
    def confirm_payment(code):
        """
        Confirm cash payment using ADMIN-GENERATED code.
        FINAL AUTHORITY — race-condition safe.
        """

        try:
            # 🔒 Lock the pending payment row
            pending = (
                PendingCashPayment.query
                .filter_by(code=code, status="PENDING")
                .with_for_update()
                .first()
            )

            if not pending:
                existing = PendingCashPayment.query.filter_by(code=code).first()
                if existing:
                    if existing.status == "PAID":
                        raise Exception("This cash code has already been used.")
                    if existing.status == "CANCELLED":
                        raise Exception("This cash payment was cancelled.")
                raise Exception("Invalid cash code.")

            cart_items = pending.cart or []
            if not cart_items:
                raise Exception("Pending payment cart is empty")

            # -----------------------------
            # 🔒 Re-check + LOCK stock rows
            # -----------------------------
            validated_items = []

            for entry in cart_items:
                barcode = entry.get("barcode")
                qty = entry.get("quantity")

                if not barcode or not qty or qty <= 0:
                    raise Exception("Invalid cart item data")

                # Lock item row
                item = (
                    Item.query
                    .filter_by(barcode=barcode)
                    .with_for_update()
                    .first()
                )

                if not item:
                    raise Exception(f"Item not found: {barcode}")

                if item.quantity <= 0:
                    raise Exception(f"Item out of stock: {item.name}")

                if qty > item.quantity:
                    raise Exception(f"Insufficient stock for {item.name}")

                validated_items.append((item, qty))

            # -----------------------------
            # Create Sales Transaction
            # -----------------------------
            transaction = SalesTransaction(user_id=pending.user_id)
            db.session.add(transaction)
            db.session.flush()  # get transaction.id

            # Deduct stock + record items
            for item, qty in validated_items:
                item.quantity -= qty

                transaction_item = SalesTransactionItem(
                    transaction_id=transaction.id,
                    item_id=item.id,
                    quantity=qty,
                    price_at_sale=item.price
                )
                db.session.add(transaction_item)

            # Mark payment as PAID only AFTER successful deduction
            pending.status = "PAID"

            db.session.commit()
            return transaction.id

        except Exception:
            db.session.rollback()
            raise

from db import db
from models.pending_wallet_payment import PendingWalletPayment
from models.ewallet import EWallet
from models.ewallet_transaction import EWalletTransaction
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item

class AdminWalletPaymentService:

    @staticmethod
    def approve(pending_id):

        # Lock pending request
        pending = (
            PendingWalletPayment.query
            .filter_by(id=pending_id, status="PENDING")
            .with_for_update()
            .first()
        )

        if not pending:
            raise Exception("Pending wallet payment not found")

        # Lock wallet
        wallet = (
            EWallet.query
            .filter_by(user_id=pending.user_id)
            .with_for_update()
            .first()
        )

        if not wallet:
            raise Exception("Wallet not found")

        total = 0
        items_cache = []

        # Validate cart
        for entry in pending.cart:
            barcode = entry.get("barcode")
            qty = entry.get("quantity")

            if not barcode or not qty:
                continue

            item = Item.query.filter_by(barcode=barcode).first()
            if not item:
                raise Exception(f"Item not found: {barcode}")

            if item.quantity < qty:
                raise Exception(f"Not enough stock for {item.name}")

            total += item.price * qty
            items_cache.append((item, qty))

        if wallet.balance < total:
            raise Exception("Insufficient wallet balance")

        # Deduct wallet
        wallet.balance -= total

        # Create sales transaction
        transaction = SalesTransaction(user_id=pending.user_id)
        db.session.add(transaction)
        db.session.flush()

        # Deduct stock
        for item, qty in items_cache:
            item.quantity -= qty
            db.session.add(SalesTransactionItem(
                transaction_id=transaction.id,
                item_id=item.id,
                quantity=qty,
                price_at_sale=item.price
            ))

        # Wallet ledger
        db.session.add(EWalletTransaction(
            wallet_id=wallet.id,
            amount=-total,
            type="payment",
            reference_id=transaction.id
        ))

        pending.status = "PAID"

        db.session.commit()
        return transaction.id

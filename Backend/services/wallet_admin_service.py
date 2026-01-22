from db import db
from models.ewallet import EWallet
from models.ewallet_transaction import EWalletTransaction
from models.user import User

class WalletAdminService:

    @staticmethod
    def cash_in(admin_id, user_id, amount):

        if amount <= 0:
            raise Exception("Invalid amount")

        user = User.query.get(user_id)
        if not user:
            raise Exception("User not found")

        wallet = (
            EWallet.query
            .filter_by(user_id=user_id)
            .with_for_update()
            .first()
        )

        if not wallet:
            wallet = EWallet(user_id=user_id)
            db.session.add(wallet)
            db.session.flush()

        wallet.balance += amount

        db.session.add(EWalletTransaction(
            wallet_id=wallet.id,
            amount=amount,
            type="topup",
            reference_id=admin_id
        ))

        db.session.commit()
        return wallet.balance

from db import db
from datetime import datetime

class EWalletTransaction(db.Model):
    __tablename__ = "ewallet_transactions"

    id = db.Column(db.Integer, primary_key=True)

    wallet_id = db.Column(
        db.Integer,
        db.ForeignKey("ewallets.id"),
        nullable=False
    )

    amount = db.Column(db.Numeric(12, 2), nullable=False)

    type = db.Column(
        db.Enum("topup", "payment", "refund", name="wallet_tx_type"),
        nullable=False
    )

    reference_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    wallet = db.relationship("EWallet", backref="transactions")

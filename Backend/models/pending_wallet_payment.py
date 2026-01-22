from db import db
from datetime import datetime

class PendingWalletPayment(db.Model):
    __tablename__ = "pending_wallet_payments"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    cart = db.Column(db.JSON, nullable=False)

    status = db.Column(
        db.Enum("PENDING", "PAID", "CANCELLED", name="wallet_payment_status"),
        default="PENDING",
        nullable=False
    )

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

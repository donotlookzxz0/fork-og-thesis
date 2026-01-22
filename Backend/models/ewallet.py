from db import db

class EWallet(db.Model):
    __tablename__ = "ewallets"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        unique=True,
        nullable=False
    )

    balance = db.Column(db.Numeric(12, 2), default=0)

    user = db.relationship("User", backref="ewallet")

    def __repr__(self):
        return f"<EWallet user={self.user_id} balance={self.balance}>"

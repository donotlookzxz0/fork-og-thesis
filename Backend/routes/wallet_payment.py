from flask import Blueprint, request, jsonify, g
from utils.auth_restrict import require_auth
from models.pending_wallet_payment import PendingWalletPayment
from models.ewallet import EWallet
from db import db

wallet_payment_bp = Blueprint("wallet_payment", __name__)

# --------------------------------------------------
# 💳 STEP 1: CUSTOMER REQUESTS WALLET PAYMENT
# --------------------------------------------------
@wallet_payment_bp.route("/start", methods=["POST"])
@require_auth(roles=("customer",))
def start_wallet_payment():
    data = request.get_json() or {}
    cart = data.get("cart")

    if not cart or not isinstance(cart, list):
        return jsonify({"error": "Cart is empty or invalid"}), 400

    # Prevent multiple pending wallet payments
    existing = PendingWalletPayment.query.filter_by(
        user_id=g.current_user.id,
        status="PENDING"
    ).first()

    if existing:
        return jsonify({
            "pending_id": existing.id,
            "message": "Wallet payment already pending approval."
        }), 200

    pending = PendingWalletPayment(
        user_id=g.current_user.id,
        cart=cart
    )

    db.session.add(pending)
    db.session.commit()

    return jsonify({
        "pending_id": pending.id,
        "message": "Wallet payment request sent. Waiting for admin approval."
    }), 201


# --------------------------------------------------
# 💳 STEP 2: CUSTOMER CHECKS STATUS
# --------------------------------------------------
@wallet_payment_bp.route("/status/<int:pending_id>", methods=["GET"])
@require_auth(roles=("customer",))
def wallet_payment_status(pending_id):
    pending = PendingWalletPayment.query.filter_by(
        id=pending_id,
        user_id=g.current_user.id
    ).first()

    if not pending:
        return jsonify({"error": "Pending wallet payment not found"}), 404

    return jsonify({
        "status": pending.status
    }), 200


# --------------------------------------------------
# 💰 GET WALLET BALANCE (CUSTOMER)
# --------------------------------------------------
@wallet_payment_bp.route("/balance", methods=["GET"])
@require_auth(roles=("customer",))
def get_wallet_balance():
    wallet = EWallet.query.filter_by(
        user_id=g.current_user.id
    ).first()

    # Wallet may not exist yet
    if not wallet:
        return jsonify({
            "username": g.current_user.username,
            "balance": 0
        }), 200

    return jsonify({
        "username": g.current_user.username,
        "balance": float(wallet.balance)
    }), 200

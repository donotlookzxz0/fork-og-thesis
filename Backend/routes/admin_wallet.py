from flask import Blueprint, jsonify, request
from utils.auth_restrict import require_auth
from models.user import User
from models.ewallet import EWallet
from models.ewallet_transaction import EWalletTransaction
from models.pending_wallet_payment import PendingWalletPayment
from services.admin_wallet_payment_service import AdminWalletPaymentService
from db import db

admin_wallet_bp = Blueprint("admin_wallet", __name__)

# --------------------------------------------------
# 📥 GET PENDING WALLET PAYMENTS (ADMIN)
# GET /payment/admin/wallet/pending
# --------------------------------------------------
@admin_wallet_bp.route("/pending", methods=["GET"])
@require_auth(roles=("admin",))
def get_pending_wallet_payments():
    results = (
        db.session.query(PendingWalletPayment, User)
        .join(User, User.id == PendingWalletPayment.user_id)
        .filter(PendingWalletPayment.status == "PENDING")
        .order_by(PendingWalletPayment.created_at.desc())
        .all()
    )

    return jsonify([
        {
            "id": pending.id,
            "user_id": pending.user_id,
            "username": user.username,
            "cart": pending.cart,
            "status": pending.status,
            "created_at": pending.created_at.isoformat()
        }
        for pending, user in results
    ]), 200


# --------------------------------------------------
# ✅ APPROVE WALLET PAYMENT (ADMIN)
# POST /payment/admin/wallet/approve/<id>
# --------------------------------------------------
@admin_wallet_bp.route("/approve/<int:pending_id>", methods=["POST"])
@require_auth(roles=("admin",))
def approve_wallet_payment(pending_id):
    try:
        AdminWalletPaymentService.approve(pending_id)
        return jsonify({
            "message": "Wallet payment approved successfully"
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": "Approval failed",
            "details": str(e)
        }), 500


# --------------------------------------------------
# 💰 GET CUSTOMER WALLET BALANCE (ADMIN)
# GET /payment/admin/wallet/balance/<id>
# --------------------------------------------------
@admin_wallet_bp.route("/balance/<int:user_id>", methods=["GET"])
@require_auth(roles=("admin",))
def get_customer_wallet_balance(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    wallet = EWallet.query.filter_by(user_id=user_id).first()

    return jsonify({
        "user_id": user.id,
        "username": user.username,
        "balance": float(wallet.balance) if wallet else 0
    }), 200


# --------------------------------------------------
# 📜 GET CUSTOMER WALLET TRANSACTIONS (ADMIN)
# GET /payment/admin/wallet/transactions/<id>
# --------------------------------------------------
@admin_wallet_bp.route("/transactions/<int:user_id>", methods=["GET"])
@require_auth(roles=("admin",))
def get_customer_wallet_transactions(user_id):
    wallet = EWallet.query.filter_by(user_id=user_id).first()
    if not wallet:
        return jsonify([]), 200

    transactions = (
        EWalletTransaction.query
        .filter_by(wallet_id=wallet.id)
        .order_by(EWalletTransaction.created_at.desc())
        .all()
    )

    return jsonify([
        {
            "id": tx.id,
            "amount": float(tx.amount),
            "type": tx.type,
            "reference_id": tx.reference_id,
            "created_at": tx.created_at.isoformat()
        }
        for tx in transactions
    ]), 200


# --------------------------------------------------
# ➕ ADMIN WALLET TOP-UP  🔥 FIXED
# POST /payment/admin/wallet/topup
# --------------------------------------------------
@admin_wallet_bp.route("/topup", methods=["POST"])
@require_auth(roles=("admin",))
def admin_wallet_topup():
    data = request.get_json() or {}

    user_id = data.get("user_id")
    amount = data.get("amount")

    # 🔒 STRICT VALIDATION
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    if amount is None:
        return jsonify({"error": "amount is required"}), 400

    try:
        amount = float(amount)
    except:
        return jsonify({"error": "Invalid amount"}), 400

    if amount <= 0:
        return jsonify({"error": "Amount must be greater than 0"}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    try:
        # get or create wallet
        wallet = EWallet.query.filter_by(user_id=user_id).first()
        if not wallet:
            wallet = EWallet(user_id=user_id, balance=0)
            db.session.add(wallet)
            db.session.flush()

        # 🔥 CRITICAL FIX — ALWAYS CAST TO FLOAT FIRST
        current_balance = float(wallet.balance or 0)
        new_balance = current_balance + amount
        wallet.balance = new_balance

        # record transaction
        tx = EWalletTransaction(
            wallet_id=wallet.id,
            amount=amount,
            type="topup",
            reference_id=None
        )

        db.session.add(tx)
        db.session.commit()

        return jsonify({
            "message": "Wallet topped up successfully",
            "user_id": user.id,
            "username": user.username,
            "new_balance": float(wallet.balance)
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "error": "Top-up failed",
            "details": str(e)
        }), 500


# --------------------------------------------------
# ❌ CANCEL WALLET PAYMENT (ADMIN)
# POST /payment/admin/wallet/cancel/<id>
# --------------------------------------------------
@admin_wallet_bp.route("/cancel/<int:pending_id>", methods=["POST"])
@require_auth(roles=("admin",))
def cancel_wallet_payment(pending_id):
    pending = PendingWalletPayment.query.filter_by(
        id=pending_id,
        status="PENDING"
    ).first()

    if not pending:
        return jsonify({ "error": "Pending wallet payment not found" }), 404

    pending.status = "CANCELLED"
    db.session.commit()

    return jsonify({
        "message": "Wallet payment cancelled"
    }), 200

from flask import Blueprint, jsonify, g
from services.admin_cash_payment_service import AdminCashPaymentService
from utils.auth_restrict import require_auth
from models.pending_cash_payment import PendingCashPayment
from models.user import User
from db import db

admin_cash_bp = Blueprint("admin_cash", __name__)

# -----------------------
# GET ALL PENDING CASH REQUESTS (ADMIN ONLY)
# -----------------------
@admin_cash_bp.route("/pending", methods=["GET"])
@require_auth(roles=("admin",))
def get_pending():
    results = (
        db.session.query(PendingCashPayment, User)
        .join(User, User.id == PendingCashPayment.user_id)
        .filter(PendingCashPayment.status == "PENDING")
        .order_by(PendingCashPayment.created_at.desc())
        .all()
    )

    return jsonify([
        {
            "id": pending.id,
            "user_id": pending.user_id,
            "username": user.username,
            "cart": pending.cart,
            "code": pending.code,
            "created_at": pending.created_at.isoformat()
        }
        for pending, user in results
    ]), 200


# -----------------------
# ADMIN GENERATES CASH CODE
# -----------------------
@admin_cash_bp.route("/generate-code/<int:pending_id>", methods=["POST"])
@require_auth(roles=("admin",))
def generate_code(pending_id):
    try:
        pending = AdminCashPaymentService.generate_code(pending_id)

        return jsonify({
            "message": "Cash code generated and sent to customer",
            "code": pending.code
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# -----------------------
# ADMIN CANCELS CASH REQUEST
# -----------------------
@admin_cash_bp.route("/cancel/<int:pending_id>", methods=["POST"])
@require_auth(roles=("admin",))
def cancel_request(pending_id):
    pending = PendingCashPayment.query.filter_by(
        id=pending_id,
        status="PENDING"
    ).first()

    if not pending:
        return jsonify({"error": "Pending cash request not found"}), 404

    pending.status = "CANCELLED"
    db.session.commit()

    return jsonify({
        "message": "Pending cash request cancelled"
    }), 200

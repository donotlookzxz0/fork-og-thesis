from flask import Blueprint, jsonify, request
from db import db

from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item

from sqlalchemy.orm import joinedload

# from utils.auth_restrict import require_auth

sales_bp = Blueprint("sales", __name__)

# --------------------------------------------------
# 🔵 GET all transactions (ACCEPT /sales AND /sales/)
# --------------------------------------------------
@sales_bp.route("", methods=["GET"])
@sales_bp.route("/", methods=["GET"])
# @require_auth(roles=("admin",))
def get_all_transactions():
    try:
        # 🔥 FORCE EAGER LOADING (NO LAZY LOAD CRASH)
        transactions = (
            SalesTransaction.query
            .options(
                joinedload(SalesTransaction.user),
                joinedload(SalesTransaction.items).joinedload(SalesTransactionItem.item)
            )
            .order_by(SalesTransaction.date.desc())
            .all()
        )

        result = []

        for t in transactions:
            # SAFE USER ACCESS
            user_id = t.user.id if t.user else None

            items = []
            for ti in t.items:
                items.append({
                    "item_id": ti.item_id,
                    "item_name": ti.item.name if ti.item else None,
                    "category": ti.item.category if ti.item else None,
                    "quantity": ti.quantity,
                    "price_at_sale": float(ti.price_at_sale)
                })

            result.append({
                "transaction_id": t.id,
                "date": t.date.isoformat(),
                "user_id": user_id,
                "items": items
            })

        return jsonify(result), 200

    except Exception as e:
        print("🔥 SALES ERROR:", e)
        return jsonify({"error": "Failed to load transactions"}), 500


# --------------------------------------------------
# 🔵 GET single transaction (ACCEPT BOTH)
# --------------------------------------------------
@sales_bp.route("/<int:id>", methods=["GET"])
@sales_bp.route("/<int:id>/", methods=["GET"])
# @require_auth(roles=("admin",))
def get_transaction(id):
    try:
        t = (
            SalesTransaction.query
            .options(
                joinedload(SalesTransaction.user),
                joinedload(SalesTransaction.items).joinedload(SalesTransactionItem.item)
            )
            .get(id)
        )

        if not t:
            return jsonify({"error": "Transaction not found"}), 404

        return jsonify({
            "transaction_id": t.id,
            "date": t.date.isoformat(),
            "user_id": t.user.id if t.user else None,
            "items": [
                {
                    "item_id": ti.item_id,
                    "item_name": ti.item.name if ti.item else None,
                    "category": ti.item.category if ti.item else None,
                    "quantity": ti.quantity,
                    "price_at_sale": float(ti.price_at_sale)
                }
                for ti in t.items
            ]
        }), 200

    except Exception as e:
        print("🔥 SALES SINGLE ERROR:", e)
        return jsonify({"error": "Failed to load transaction"}), 500


# --------------------------------------------------
# 🟢 CREATE transaction (ACCEPT /sales AND /sales/)
# --------------------------------------------------
@sales_bp.route("", methods=["POST"])
@sales_bp.route("/", methods=["POST"])
# @require_auth()
def create_transaction():
    data = request.get_json() or {}
    cart_items = data.get("items", [])
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    if not cart_items:
        return jsonify({"error": "No items provided"}), 400

    transaction = SalesTransaction(user_id=user_id)
    db.session.add(transaction)
    db.session.flush()

    for entry in cart_items:
        item_id = entry.get("item_id")
        qty = entry.get("quantity")

        if not item_id or not qty:
            return jsonify({"error": "item_id and quantity required"}), 400

        item = Item.query.get(item_id)
        if not item:
            return jsonify({"error": f"Item {item_id} not found"}), 400

        if item.quantity < qty:
            return jsonify({"error": f"Not enough stock for {item.name}"}), 400

        item.quantity -= qty

        db.session.add(SalesTransactionItem(
            transaction=transaction,
            item=item,
            quantity=qty,
            price_at_sale=item.price
        ))

    db.session.commit()

    return jsonify({
        "message": "Transaction recorded",
        "transaction_id": transaction.id
    }), 201


# --------------------------------------------------
# 🔵 UPDATE transaction (ACCEPT BOTH)
# --------------------------------------------------
@sales_bp.route("/<int:id>", methods=["PUT"])
@sales_bp.route("/<int:id>/", methods=["PUT"])
# @require_auth(roles=("admin",))
def update_transaction(id):
    t = SalesTransaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404

    data = request.get_json() or {}
    new_items = data.get("items", [])

    if not new_items:
        return jsonify({"error": "No items provided"}), 400

    # restore stock
    for ti in t.items:
        if ti.item:
            ti.item.quantity += ti.quantity

    SalesTransactionItem.query.filter_by(transaction_id=t.id).delete()

    for entry in new_items:
        item_id = entry.get("item_id")
        qty = entry.get("quantity")

        item = Item.query.get(item_id)
        if not item or item.quantity < qty:
            return jsonify({"error": "Invalid item or insufficient stock"}), 400

        item.quantity -= qty

        db.session.add(SalesTransactionItem(
            transaction=t,
            item=item,
            quantity=qty,
            price_at_sale=item.price
        ))

    db.session.commit()
    return jsonify({"message": "Transaction updated"}), 200


# --------------------------------------------------
# 🔴 DELETE transaction (ACCEPT BOTH)
# --------------------------------------------------
@sales_bp.route("/<int:id>", methods=["DELETE"])
@sales_bp.route("/<int:id>/", methods=["DELETE"])
# @require_auth(roles=("admin",))
def delete_transaction(id):
    t = SalesTransaction.query.get(id)
    if not t:
        return jsonify({"error": "Transaction not found"}), 404

    for ti in t.items:
        if ti.item:
            ti.item.quantity += ti.quantity

    db.session.delete(t)
    db.session.commit()

    return jsonify({"message": "Transaction deleted"}), 200

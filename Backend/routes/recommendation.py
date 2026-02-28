from flask import Blueprint, jsonify, g
from ml.recommender.inference import recommend_for_user
from utils.auth_restrict import require_auth
from models.user import User
from ml.recommender.trainer import retrain_model
from flask import current_app

recommendations_bp = Blueprint("recommendations_bp", __name__)

_training_in_progress = False

# GET recommendations for current logged in user in React
@recommendations_bp.route("/recommendations/me", methods=["GET"])
@require_auth(roles=("customer",))
def get_my_recommendations():
    user_id = g.current_user.id
    items = recommend_for_user(user_id)

    return jsonify({
        "user_id": user_id,
        "recommendations": [
            {
                "id": i.id,
                "barcode": i.barcode,
                "name": i.name,
                "category": i.category,
                "price": float(i.price),
                "quantity": i.quantity
            }
            for i in items
        ]
    }), 200

# GET recommendations for a SPECIFIC user (admin only)
@recommendations_bp.route("/recommendations/<int:user_id>", methods=["GET"])
@require_auth(roles=("admin",))
def get_user_recommendations(user_id):
    items = recommend_for_user(user_id)
    return jsonify({
        "user_id": user_id,
        "recommendations": [
            {
                "id": i.id,
                "barcode": i.barcode,
                "name": i.name,
                "category": i.category,
                "price": float(i.price)
            }
            for i in items
        ]
    }), 200

# GET recommendations for ALL users
@recommendations_bp.route("/recommendations", methods=["GET"])
@require_auth(roles=("admin",))
def get_all_recommendations():
    users = User.query.all()
    if not users:
        return jsonify({"message": "No users found"}), 200

    all_recommendations = []

    for user in users:
        items = recommend_for_user(user.id)

        all_recommendations.append({
            "user_id": user.id,
            "recommendations": [
                {
                    "id": i.id,
                    "barcode": i.barcode,
                    "name": i.name,
                    "category": i.category,
                    "price": float(i.price)
                }
                for i in items
            ]
        })

    return jsonify(all_recommendations), 200

@recommendations_bp.route("/recommendations/train", methods=["POST"])
@require_auth(roles=("admin",))
def train_recommender():
    global _training_in_progress

    if _training_in_progress:
        return jsonify({
            "success": False,
            "message": "Training already in progress"
        }), 409

    try:
        _training_in_progress = True

        result = retrain_model()

        return jsonify({
            "success": True,
            "message": "Training completed successfully",
            "logs": result["logs"],
            "rmse": result["rmse"],
            "mse": result["mse"]
        }), 200

    except Exception as e:
        current_app.logger.exception("Training failed")
        return jsonify({
            "success": False,
            "message": "Training failed",
            "error": str(e)
        }), 500

    finally:
        _training_in_progress = False
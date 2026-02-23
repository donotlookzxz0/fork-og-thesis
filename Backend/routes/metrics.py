from flask import Blueprint, jsonify

from ml.time_series_forecast import run_time_series_forecast
from ml.item_movement_forecast import run_item_movement_forecast
from ml.stockout_risk_forecast import run_stockout_risk_forecast

# from utils.auth_restrict import require_auth

metrics_bp = Blueprint("metrics_bp", __name__)


# =================================================
# FORECAST METRICS
# =================================================

@metrics_bp.route("/forecast", methods=["GET"])
# @require_auth()
def get_forecast_metrics():
    try:
        data = run_time_series_forecast()

        if data is None:
            return jsonify({
                "success": False,
                "message": "Not enough data"
            }), 400

        return jsonify({
            "success": True,
            "metrics": data.get("metrics", {})
        }), 200

    except Exception as e:
        print("🔥 FORECAST METRICS ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =================================================
# ITEM MOVEMENT METRICS (STRUCTURE READY)
# =================================================

@metrics_bp.route("/item-movement", methods=["GET"])
# @require_auth()
def get_item_movement_metrics():
    try:
        data = run_item_movement_forecast()

        if not data:
            return jsonify({
                "success": False,
                "message": "Not enough data"
            }), 400

        return jsonify({
            "success": True,
            "metrics": data.get("metrics", {})
        }), 200

    except Exception as e:
        print("🔥 ITEM MOVEMENT METRICS ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =================================================
# STOCKOUT RISK METRICS (STRUCTURE READY)
# =================================================

@metrics_bp.route("/stockout-risk", methods=["GET"])
# @require_auth()
def get_stockout_risk_metrics():
    try:
        data = run_stockout_risk_forecast()

        if not data:
            return jsonify({
                "success": False,
                "message": "Not enough data"
            }), 400

        return jsonify({
            "success": True,
            "metrics": data.get("metrics", {})
        }), 200

    except Exception as e:
        print("🔥 STOCKOUT METRICS ERROR:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
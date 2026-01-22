# app.py

import os
from flask import Flask, request
from flask_cors import CORS

from db import db
from urls import register_routes

app = Flask(__name__)

# --------------------------------------------------
#  SECURITY / COOKIE CONFIG (FROM .ENV)
# --------------------------------------------------
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "None"   # REQUIRED for cross-origin cookies
app.config["SESSION_COOKIE_SECURE"] = True       # MUST be TRUE for HTTPS

# --------------------------------------------------
#  CORS ( FIXED FOR COOKIES)
# --------------------------------------------------
CORS(
    app,
    supports_credentials=True,
    origins=[
        os.getenv("CORS_ORIGIN_MOBILE"),
        os.getenv("CORS_ORIGIN_PC"),
    ],
    allow_headers=["Content-Type", "Authorization"],
)

# --------------------------------------------------
#  GLOBAL PREFLIGHT HANDLER
# --------------------------------------------------
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return "", 200

# --------------------------------------------------
# 🗄 DATABASE (POSTGRESQL — FROM .ENV)
# --------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_pre_ping": True,
    "pool_recycle": 280,
}

db.init_app(app)

# --------------------------------------------------
#  AUTO CREATE TABLES (SAFE FOR GUNICORN)
# --------------------------------------------------
with app.app_context():
    db.create_all()

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
register_routes(app)

# --------------------------------------------------
# ROOT CHECK
# --------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return {"message": "Flask API running successfully"}

# --------------------------------------------------
#  LOCAL DEV ONLY
# --------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        use_reloader=False,
    )

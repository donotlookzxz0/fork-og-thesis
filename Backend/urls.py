from routes import (
    items_bp,
    sales_bp,
    user_routes,
    payment_bp,
    cash_payment_bp,
    admin_cash_bp,
    ml_bp,
    metrics_bp,
    recommendations_bp,
)

from routes.wallet_payment import wallet_payment_bp
from routes.admin_wallet import admin_wallet_bp


def register_routes(app):
    app.register_blueprint(items_bp, url_prefix="/items")
    app.register_blueprint(sales_bp, url_prefix="/sales")
    app.register_blueprint(user_routes, url_prefix="/users")
    app.register_blueprint(ml_bp, url_prefix="/ml")
    app.register_blueprint(metrics_bp, url_prefix="/metrics")
    app.register_blueprint(recommendations_bp, url_prefix="/")

    app.register_blueprint(payment_bp, url_prefix="/payment")

    app.register_blueprint(cash_payment_bp, url_prefix="/payment/cash")
    app.register_blueprint(admin_cash_bp, url_prefix="/payment/admin/cash")

    app.register_blueprint(wallet_payment_bp, url_prefix="/payment/wallet")
    app.register_blueprint(admin_wallet_bp, url_prefix="/payment/admin/wallet")

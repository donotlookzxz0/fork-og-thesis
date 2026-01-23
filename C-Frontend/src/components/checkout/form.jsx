import "bootstrap/dist/css/bootstrap.min.css";
import './styles.css'

export default function CheckoutForm({
  cart,
  totalPrice,
  paymentMethod,
  setPaymentMethod,
  paymentLocked,
  waitingForAdmin,
  waitingWalletApproval,
  cashCode,
  setCashCode,
  onConfirmCash,
  onCancelCash,
  onPlaceOrder,
  navigate,
  pendingId,
}) {
  return (
    <div className="container mt-5">
      <div className="card shadow-sm p-4 checkout">
        <button
          onClick={() => navigate("/cart")}
          className="btn btn-link p-0 mb-3"
        >
          ← Back to Cart
        </button>

        <h2 className="mb-4">Checkout</h2>

        <ul className="list-group mb-3">
          {cart.map(item => (
            <li key={item.barcode} className="list-group-item d-flex justify-content-between">
              {item.name} (x{item.quantity})
              <span>₱{item.price * item.quantity}</span>
            </li>
          ))}
        </ul>

        <p className="fw-bold">Total: ₱{totalPrice}</p>

        {/* Payment Method Radios */}
        <h5 className="mt-4">Payment Method</h5>

        {["cash", "wallet", "gcash"].map(method => (
          <div className="form-check form-check-inline" key={method}>
            <input
              className="form-check-input"
              type="radio"
              value={method}
              checked={paymentMethod === method}
              disabled={paymentLocked}
              onChange={() => setPaymentMethod(method)}
              style={{ accentColor: "#ea1e25" }}
            />
            <label className="form-check-label text-capitalize">
              {method}
            </label>
          </div>
        ))}

        {waitingForAdmin && (
          <p className="text-warning mt-3">Waiting for admin to generate your cash code…</p>
        )}

        {waitingWalletApproval && (
          <p className="text-warning mt-3">Waiting for admin to approve wallet payment…</p>
        )}

        {paymentMethod === "cash" && cashCode && (
          <>
            <input
              className="form-control mt-3"
              value={cashCode}
              onChange={e => setCashCode(e.target.value)}
            />
            <button className="btn btn-success mt-3 w-100" onClick={onConfirmCash}>
              Confirm Cash Payment
            </button>
            <button className="btn btn-outline-danger mt-2 w-100" onClick={onCancelCash}>
              Cancel Cash Request
            </button>
          </>
        )}

        {/* MAIN BUTTON — unchanged */}
        <button
          className="btn btn-primary w-100 mt-3 button"
          onClick={onPlaceOrder}
          disabled={paymentLocked}
        >
          {paymentMethod === "gcash"
            ? "Pay with GCash"
            : paymentMethod === "wallet"
            ? waitingWalletApproval
              ? "Waiting for Wallet Approval"
              : "Pay with Wallet"
            : pendingId
            ? "Cash Payment Requested"
            : "Request Cash Payment"}
        </button>
      </div>
    </div>
  );
}
import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { useNavigate } from "react-router-dom";
import api from "../api/axios";

function Checkout({ cart, setCart }) {
  const [paymentMethod, setPaymentMethod] = useState("gcash");

  // CASH
  const [cashCode, setCashCode] = useState("");
  const [pendingId, setPendingId] = useState(null);
  const [waitingForAdmin, setWaitingForAdmin] = useState(false);

  // WALLET
  const [walletPendingId, setWalletPendingId] = useState(null);
  const [waitingWalletApproval, setWaitingWalletApproval] = useState(false);

  const [isPlacingOrder, setIsPlacingOrder] = useState(false);
  const navigate = useNavigate();

  const totalPrice = cart.reduce(
    (sum, item) => sum + Number(item.price) * item.quantity,
    0
  );

  /* -----------------------
     POLL FOR ADMIN CASH CODE
  ----------------------- */
  useEffect(() => {
    if (!pendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(`payment/cash/status/${pendingId}`);
        const data = res.data;

        if (data.status === "CANCELLED") {
          alert("Your cash payment request was cancelled.");
          setPendingId(null);
          setWaitingForAdmin(false);
          setCashCode("");
          clearInterval(interval);
          return;
        }

        if (data.code) {
          setCashCode(data.code);
          setWaitingForAdmin(false);
          clearInterval(interval);
          alert(`Your cash code is: ${data.code}`);
        }
      } catch (err) {
        console.error("Failed to poll cash status", err);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [pendingId]);

  /* -----------------------
     POLL FOR WALLET STATUS
  ----------------------- */
  useEffect(() => {
    if (!walletPendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(
          `payment/wallet/status/${walletPendingId}`
        );
        const data = res.data;

        if (data.status === "PAID") {
          clearInterval(interval);
          setCart([]);
          localStorage.removeItem("cart");
          navigate("/success");
        }

        if (data.status === "CANCELLED") {
          clearInterval(interval);
          alert("Wallet payment was cancelled.");
          setWalletPendingId(null);
          setWaitingWalletApproval(false);
        }
      } catch (err) {
        console.error("Failed to poll wallet status", err);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [walletPendingId, navigate, setCart]);

  /* -----------------------
     PLACE ORDER
  ----------------------- */
  const handlePlaceOrder = async () => {
    if (
      isPlacingOrder ||
      pendingId ||
      walletPendingId
    )
      return;

    if (cart.length === 0) {
      alert("Your cart is empty!");
      return;
    }

    setIsPlacingOrder(true);

    try {
      if (paymentMethod === "gcash") {
        const intentRes = await api.post("payment/intent", {
          amount: totalPrice * 100,
          currency: "PHP",
        });

        const checkoutRes = await api.post("payment/checkout", {
          payment_intent_id: intentRes.data.id,
          success_url: `${window.location.origin}/success`,
          cancel_url: `${window.location.origin}/cancel`,
          cart,
        });

        window.location.href = checkoutRes.data.checkoutUrl;

      } else if (paymentMethod === "cash") {
        const res = await api.post("payment/cash/start", { cart });
        setPendingId(res.data.pending_id);
        setWaitingForAdmin(true);
        alert("Cash payment requested. Waiting for admin approval.");

      } else if (paymentMethod === "wallet") {
        const res = await api.post("payment/wallet/start", { cart });
        setWalletPendingId(res.data.pending_id);
        setWaitingWalletApproval(true);
        alert("Wallet payment requested. Waiting for admin approval.");
      }
    } catch (err) {
      console.error(err);
      alert("Failed to place order.");
    } finally {
      setIsPlacingOrder(false);
    }
  };

  /* -----------------------
     CONFIRM CASH PAYMENT
  ----------------------- */
  const handleConfirmCash = async () => {
    if (!cashCode || !pendingId) {
      alert("Waiting for admin-generated cash code.");
      return;
    }

    try {
      const res = await api.post("payment/cash/confirm", { code: cashCode });
      alert(res.data.message);
      setCart([]);
      localStorage.removeItem("cart");
      navigate("/success");
    } catch (err) {
      console.error(err);
      alert("Error confirming cash payment.");
    }
  };

  /* -----------------------
     CANCEL CASH PAYMENT
  ----------------------- */
  const handleCancelCash = async () => {
    if (!pendingId) return;

    if (!window.confirm("Are you sure you want to cancel this cash payment?"))
      return;

    try {
      await api.post(`payment/cash/cancel/${pendingId}`);
      alert("Cash payment cancelled.");
      setPendingId(null);
      setWaitingForAdmin(false);
      setCashCode("");
    } catch (err) {
      console.error(err);
      alert("Failed to cancel cash payment.");
    }
  };

  const paymentLocked =
    pendingId ||
    walletPendingId ||
    waitingForAdmin ||
    waitingWalletApproval;

  return (
    <div className="container mt-5">
      <div className="card shadow-sm p-4">
        <button
          onClick={() => navigate("/items")}
          style={{
            background: "transparent",
            border: "none",
            padding: 0,
            marginBottom: 16,
            color: "#113F67",
            fontWeight: 600,
            fontSize: 14,
            cursor: "pointer",
          }}
        >
          ← Back to Cart
        </button>

        <h2 className="mb-4">Checkout</h2>

        <div className="mb-4">
          <h5>Your Items:</h5>
          <ul className="list-group mb-3">
            {cart.map((item) => (
              <li
                key={item.barcode}
                className="list-group-item d-flex justify-content-between"
              >
                {item.name} (x{item.quantity})
                <span>₱{item.price * item.quantity}</span>
              </li>
            ))}
          </ul>
          <p className="fw-bold">Total: ₱{totalPrice}</p>
        </div>

        <div className="mb-4">
          <h5>Payment Method:</h5>
          <select
            className="form-control"
            value={paymentMethod}
            disabled={paymentLocked}
            onChange={(e) => setPaymentMethod(e.target.value)}
          >
            <option value="gcash">GCash</option>
            <option value="cash">Cash</option>
            <option value="wallet">Wallet</option>
          </select>
        </div>

        {waitingForAdmin && (
          <p className="text-warning">
            Waiting for admin to generate your cash code…
          </p>
        )}

        {waitingWalletApproval && (
          <p className="text-warning">
            Waiting for admin to approve wallet payment…
          </p>
        )}

        {paymentMethod === "cash" && cashCode && (
          <>
            <input
              className="form-control"
              value={cashCode}
              onChange={(e) => setCashCode(e.target.value)}
            />
            <button
              className="btn btn-success mt-3 w-100"
              onClick={handleConfirmCash}
            >
              Confirm Cash Payment
            </button>
            <button
              className="btn btn-outline-danger mt-2 w-100"
              onClick={handleCancelCash}
            >
              Cancel Cash Request
            </button>
          </>
        )}

        <button
          className="btn btn-primary w-100 mt-3"
          onClick={handlePlaceOrder}
          disabled={paymentLocked || isPlacingOrder}
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

export default Checkout;

import { useEffect, useState } from "react";
import api from "../../api/axios";

export function useCash({ cart, setCart, navigate, totalPrice }) {
  const [cashCode, setCashCode] = useState("");
  const [pendingId, setPendingId] = useState(null);
  const [waitingForAdmin, setWaitingForAdmin] = useState(false);

  useEffect(() => {
    if (!pendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(`/payment/cash/status/${pendingId}`);
        const data = res.data;

        if (data.status === "CANCELLED") {
          alert("Your cash payment request was cancelled.");
          resetCash();
          clearInterval(interval);
        }

        if (data.code) {
          setCashCode(data.code);
          setWaitingForAdmin(false);
          alert(`Your cash code is: ${data.code}`);
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Failed to poll cash status", err);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [pendingId]);

  // 🔒 START CASH PAYMENT — WITH PROPER ERROR HANDLING
  const startCashPayment = async () => {
    try {
      const res = await api.post("/payment/cash/start", { cart });

      setPendingId(res.data.pending_id);
      setWaitingForAdmin(true);
      alert("Cash payment requested. Waiting for admin approval.");

    } catch (err) {
      console.error("Cash payment failed:", err);

      const backendMessage =
        err.response?.data?.message ||
        err.response?.data?.error ||
        "";

      let userMessage = "Payment failed. Please try again.";

      // 🛑 Friendly stock / cart messages
      if (
        backendMessage.toLowerCase().includes("out of stock") ||
        backendMessage.toLowerCase().includes("insufficient stock")
      ) {
        userMessage = "One or more items in your cart are out of stock. Please update your cart.";
      } 
      else if (
        backendMessage.toLowerCase().includes("item not found") ||
        backendMessage.toLowerCase().includes("invalid cart")
      ) {
        userMessage = "One or more items in your cart are invalid. Please review your cart.";
      } 
      else if (backendMessage) {
        // Clean backend message (ex: wallet balance, etc.)
        userMessage = backendMessage;
      }

      alert(userMessage);
    }
  };

  const confirmCash = async () => {
    if (!cashCode || !pendingId) {
      alert("Waiting for admin-generated cash code.");
      return;
    }

    try {
      const res = await api.post("/payment/cash/confirm", { code: cashCode });

      alert(res.data.message);
      setCart([]);
      localStorage.removeItem("cart");
      navigate("/success", {
        state: { totalPrice }
      });

    } catch (err) {
      console.error("Cash confirmation failed:", err);

      const message =
        err.response?.data?.message ||
        err.response?.data?.error ||
        "Cash confirmation failed";

      alert(message);
    }
  };

  const cancelCash = async () => {
    if (!pendingId) return;
    if (!window.confirm("Are you sure you want to cancel this cash payment?")) return;

    try {
      await api.post(`/payment/cash/cancel/${pendingId}`);
      alert("Cash payment cancelled.");
      resetCash();

    } catch (err) {
      console.error("Failed to cancel cash payment:", err);
      alert("Failed to cancel cash payment. Please try again.");
    }
  };

  const resetCash = () => {
    setPendingId(null);
    setWaitingForAdmin(false);
    setCashCode("");
  };

  return {
    cashCode,
    setCashCode,
    pendingId,
    waitingForAdmin,
    startCashPayment,
    confirmCash,
    cancelCash,
  };
}

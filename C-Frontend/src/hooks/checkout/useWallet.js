import { useEffect, useState } from "react";
import api from "../../api/axios";

export function useWallet({ cart, setCart, navigate, totalPrice }) {
  const [walletPendingId, setWalletPendingId] = useState(null);
  const [waitingWalletApproval, setWaitingWalletApproval] = useState(false);

  useEffect(() => {
    if (!walletPendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(`/payment/wallet/status/${walletPendingId}`);
        const data = res.data;

        if (data.status === "PAID") {
          clearInterval(interval);
          setCart([]);
          localStorage.removeItem("cart");
          navigate("/success", {
            state: { totalPrice }
          });
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
  }, [walletPendingId]);

  // 🔒 START WALLET PAYMENT — WITH PROPER ERROR HANDLING
  const startWalletPayment = async () => {
    try {
      const res = await api.post("/payment/wallet/start", { cart });

      setWalletPendingId(res.data.pending_id);
      setWaitingWalletApproval(true);
      alert("Wallet payment requested. Waiting for admin approval.");

    } catch (err) {
      console.error("Wallet payment failed:", err);

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
      else if (
        backendMessage.toLowerCase().includes("wallet balance") ||
        backendMessage.toLowerCase().includes("insufficient wallet")
      ) {
        userMessage = "Your wallet balance is insufficient for this purchase.";
      } 
      else if (backendMessage) {
        // Clean backend business message
        userMessage = backendMessage;
      }

      alert(userMessage);
    }
  };

  return {
    walletPendingId,
    waitingWalletApproval,
    startWalletPayment,
  };
}

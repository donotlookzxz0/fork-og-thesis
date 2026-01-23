import { useEffect, useState } from "react";
import api from "../../api/axios";

export function useWallet({ cart, setCart, navigate }) {
  const [walletPendingId, setWalletPendingId] = useState(null);
  const [waitingWalletApproval, setWaitingWalletApproval] = useState(false);

  useEffect(() => {
    if (!walletPendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(`payment/wallet/status/${walletPendingId}`);
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
  }, [walletPendingId]);

  const startWalletPayment = async () => {
    const res = await api.post("payment/wallet/start", { cart });
    setWalletPendingId(res.data.pending_id);
    setWaitingWalletApproval(true);
    alert("Wallet payment requested. Waiting for admin approval.");
  };

  return {
    walletPendingId,
    waitingWalletApproval,
    startWalletPayment,
  };
}
import { useEffect, useState } from "react";
import api from "../../api/axios";

export function useCash({ cart, setCart, navigate }) {
  const [cashCode, setCashCode] = useState("");
  const [pendingId, setPendingId] = useState(null);
  const [waitingForAdmin, setWaitingForAdmin] = useState(false);

  useEffect(() => {
    if (!pendingId) return;

    const interval = setInterval(async () => {
      try {
        const res = await api.get(`payment/cash/status/${pendingId}`);
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

  const startCashPayment = async () => {
    const res = await api.post("payment/cash/start", { cart });
    setPendingId(res.data.pending_id);
    setWaitingForAdmin(true);
    alert("Cash payment requested. Waiting for admin approval.");
  };

  const confirmCash = async () => {
    if (!cashCode || !pendingId) {
      alert("Waiting for admin-generated cash code.");
      return;
    }

    const res = await api.post("payment/cash/confirm", { code: cashCode });
    alert(res.data.message);
    setCart([]);
    localStorage.removeItem("cart");
    navigate("/success");
  };

  const cancelCash = async () => {
    if (!pendingId) return;
    if (!window.confirm("Are you sure you want to cancel this cash payment?")) return;

    await api.post(`payment/cash/cancel/${pendingId}`);
    alert("Cash payment cancelled.");
    resetCash();
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
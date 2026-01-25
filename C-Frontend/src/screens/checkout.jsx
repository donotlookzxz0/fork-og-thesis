import { useState } from "react";
import { useNavigate } from "react-router-dom";

import CheckoutForm from "../components/checkout/form";
import { useCash } from "../hooks/checkout/useCash";
import { useWallet } from "../hooks/checkout/useWallet";

export default function Checkout({ cart, setCart }) {
  const navigate = useNavigate();

  // Cash FIRST (default)
  const [paymentMethod, setPaymentMethod] = useState("cash");

  const totalPrice = cart.reduce(
    (sum, item) => sum + Number(item.price) * item.quantity,
    0
  );

  const cash = useCash({ cart, setCart, navigate });
  const wallet = useWallet({ cart, setCart, navigate });
 
  const paymentLocked =
    cash.pendingId ||
    wallet.walletPendingId ||
    cash.waitingForAdmin ||
    wallet.waitingWalletApproval;

  const handlePlaceOrder = async () => {
    if (cart.length === 0) return alert("Your cart is empty!");

    if (paymentMethod === "cash") return cash.startCashPayment();
    if (paymentMethod === "wallet") return wallet.startWalletPayment();
  };

  return (
    <CheckoutForm
      cart={cart}
      totalPrice={totalPrice}
      paymentMethod={paymentMethod}
      setPaymentMethod={setPaymentMethod}
      paymentLocked={paymentLocked}
      waitingForAdmin={cash.waitingForAdmin}
      waitingWalletApproval={wallet.waitingWalletApproval}
      cashCode={cash.cashCode}
      setCashCode={cash.setCashCode}
      onConfirmCash={cash.confirmCash}
      onCancelCash={cash.cancelCash}
      onPlaceOrder={handlePlaceOrder}
      pendingId={cash.pendingId}
      navigate={navigate}
    />
  );
}
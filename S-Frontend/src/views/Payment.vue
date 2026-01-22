<script setup>
import { onMounted } from "vue";
import { useCashPayment } from "../composables/useCashPayment";
import { useWalletPayment } from "../composables/useWalletPayment";
import { CashAPI } from "../services/cashApi";
import { WalletAPI } from "../services/walletApi";

/* -----------------------
   CASH PAYMENTS
----------------------- */
const {
  pending: cashPending,
  fetchPending: fetchCashPending,
  generateCode,
} = useCashPayment();

/* -----------------------
   WALLET PAYMENTS
----------------------- */
const {
  pending: walletPending,
  fetchPending: fetchWalletPending,
  approve,
} = useWalletPayment();

onMounted(() => {
  fetchCashPending();
  fetchWalletPending();
});

/* -----------------------
   GENERATE CASH CODE
----------------------- */
const handleGenerate = async (id) => {
  try {
    const data = await generateCode(id);
    alert(`Cash code generated: ${data.code}`);
    fetchCashPending();
  } catch (err) {
    alert(err?.message || "Failed to generate code");
  }
};

/* -----------------------
   APPROVE WALLET PAYMENT
----------------------- */
const handleApproveWallet = async (id) => {
  const confirmed = confirm(
    "Approve this wallet payment? This will deduct the user's wallet balance."
  );
  if (!confirmed) return;

  try {
    await approve(id);
    alert("Wallet payment approved");
    fetchWalletPending();
  } catch (err) {
    console.error(err);
    alert(err?.response?.data?.error || "Wallet approval failed");
  }
};

/* -----------------------
   CANCEL WALLET PAYMENT (NEW)
----------------------- */
const handleCancelWallet = async (id) => {
  const confirmed = confirm("Reject this wallet payment?");
  if (!confirmed) return;

  try {
    await WalletAPI.cancel(id);
    alert("Wallet payment rejected");
    fetchWalletPending();
  } catch (err) {
    console.error(err);
    alert(err?.response?.data?.error || "Wallet cancel failed");
  }
};

const getCartTotal = (cart = []) =>
  cart.reduce((sum, item) => sum + item.price * item.quantity, 0);

/* -----------------------
   CANCEL CASH REQUEST
----------------------- */
const cancelRequest = async (id) => {
  const confirmed = confirm(
    "Are you sure you want to cancel this cash payment?"
  );
  if (!confirmed) return;

  const reason = prompt(
    "Reason for cancelling this cash request? (optional)"
  );

  try {
    await CashAPI.cancel(id, reason);
    alert("Cash payment cancelled");
    fetchCashPending();
  } catch (err) {
    console.error(err);
    alert(err?.response?.data?.error || "Cancel failed");
  }
};
</script>

<template>
  <!-- =======================
       CASH PAYMENTS
  ======================= -->
  <h1>Cash Payments</h1>

  <div
    v-for="p in cashPending"
    :key="`cash-${p.id}`"
    style="border: 1px solid #ddd; padding: 12px; margin-bottom: 10px; width: 526px;"
  >
    <p>
      <strong>User:</strong> {{ p.username }}
      <span style="color: #888;">(ID: {{ p.user_id }})</span>
    </p>

    <div v-if="p.cart && p.cart.length">
      <strong>Cart Items:</strong>

      <table style="width: 100%; margin-top: 6px;">
        <tr
          v-for="(item, index) in p.cart"
          :key="index"
          style="border-top: 1px solid #ddd;"
        >
          <td>{{ item.name }}</td>
          <td>x{{ item.quantity }}</td>
          <td>₱{{ item.price }}</td>
        </tr>
      </table>

      <div style="margin-top: 8px; font-weight: bold;">
        Total: ₱{{ getCartTotal(p.cart) }}
      </div>
    </div>

    <p v-if="p.code">
      <strong>Cash Code:</strong>
      <span style="font-weight: bold; color: #2c7be5;">
        {{ p.code }}
      </span>
    </p>

    <button @click="handleGenerate(p.id)" :disabled="!!p.code">
      {{ p.code ? "Code Generated" : "Generate 6-Digit Code" }}
    </button>

    <button
      @click="cancelRequest(p.id)"
      style="margin-left: 10px; color: red"
    >
      Cancel
    </button>
  </div>

  <hr />

  <!-- =======================
       WALLET PAYMENTS
  ======================= -->
  <h1>Wallet Payments</h1>

  <div
    v-for="p in walletPending"
    :key="`wallet-${p.id}`"
    style="border: 1px solid #ddd; padding: 12px; margin-bottom: 10px; width: 526px;"
  >
    <p>
      <strong>User:</strong> {{ p.username }}
      <span style="color: #888;">(ID: {{ p.user_id }})</span>
    </p>

    <div v-if="p.cart && p.cart.length">
      <strong>Cart Items:</strong>

      <table style="width: 100%; margin-top: 6px;">
        <tr
          v-for="(item, index) in p.cart"
          :key="index"
          style="border-top: 1px solid #ddd;"
        >
          <td>{{ item.name }}</td>
          <td>x{{ item.quantity }}</td>
          <td>₱{{ item.price }}</td>
        </tr>
      </table>

      <div style="margin-top: 8px; font-weight: bold;">
        Total: ₱{{ getCartTotal(p.cart) }}
      </div>
    </div>

    <button @click="handleApproveWallet(p.id)">
      Approve Wallet Payment
    </button>

    <button
      @click="handleCancelWallet(p.id)"
      style="margin-left: 10px; color: red"
    >
      Reject Wallet Payment
    </button>
  </div>
</template>

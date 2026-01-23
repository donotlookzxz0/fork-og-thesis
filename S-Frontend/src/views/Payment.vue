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
   CANCEL WALLET PAYMENT
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
  <div style="display: flex; gap: 40px; align-items: flex-start; padding: 20px;">

    <!-- =======================
         CASH PAYMENTS
    ======================= -->
    <div style="flex: 1;">
      <h1 style="margin-bottom: 16px;">💵 Cash Payments</h1>

      <div
        v-for="p in cashPending"
        :key="`cash-${p.id}`"
        style="
          background: white;
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 16px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.06);
          border-left: 6px solid #4f46e5;
        "
      >
        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong style="font-size: 16px;">{{ p.username }}</strong>
            <span style="color:#888; margin-left:6px;">ID: {{ p.user_id }}</span>
          </div>

          <span
            style="
              background:#eef2ff;
              color:#4338ca;
              padding:4px 12px;
              border-radius:20px;
              font-size:12px;
              font-weight:600;
            "
          >
            CASH
          </span>
        </div>

        <!-- Cart -->
        <div v-if="p.cart && p.cart.length" style="margin-top: 14px;">
          <table style="width:100%; border-collapse: collapse;">
            <tr
              v-for="(item, index) in p.cart"
              :key="index"
              style="border-bottom: 1px solid #eee;"
            >
              <td style="padding:6px 0;">{{ item.name }}</td>
              <td style="width:60px;">x{{ item.quantity }}</td>
              <td style="width:90px; text-align:right;">₱{{ item.price }}</td>
            </tr>
          </table>

          <div
            style="
              margin-top: 10px;
              display:flex;
              justify-content:space-between;
              font-weight:bold;
              font-size:15px;
            "
          >
            <span>Total</span>
            <span>₱{{ getCartTotal(p.cart) }}</span>
          </div>
        </div>

        <!-- Code -->
        <div v-if="p.code" style="margin-top: 14px;">
          <span style="color:#666;">Cash Code</span>
          <div
            style="
              margin-top:4px;
              font-size:24px;
              font-weight:700;
              letter-spacing:3px;
              color:#2563eb;
            "
          >
            {{ p.code }}
          </div>
        </div>

        <!-- Actions -->
        <div style="margin-top: 16px; display:flex; gap:10px;">
          <button
            @click="handleGenerate(p.id)"
            :disabled="!!p.code"
            style="
              background:#4f46e5;
              color:white;
              border:none;
              padding:9px 16px;
              border-radius:8px;
              cursor:pointer;
              font-weight:600;
            "
          >
            {{ p.code ? "Code Generated" : "Generate Code" }}
          </button>

          <button
            @click="cancelRequest(p.id)"
            style="
              background:#fff;
              color:#dc2626;
              border:1px solid #dc2626;
              padding:9px 16px;
              border-radius:8px;
              cursor:pointer;
              font-weight:600;
            "
          >
            Cancel
          </button>
        </div>
      </div>
    </div>

    <!-- =======================
         WALLET PAYMENTS
    ======================= -->
    <div style="flex: 1;">
      <h1 style="margin-bottom: 16px;">👛 Wallet Payments</h1>

      <div
        v-for="p in walletPending"
        :key="`wallet-${p.id}`"
        style="
          background: white;
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 16px;
          box-shadow: 0 2px 10px rgba(0,0,0,0.06);
          border-left: 6px solid #16a34a;
        "
      >
        <!-- Header -->
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <strong style="font-size: 16px;">{{ p.username }}</strong>
            <span style="color:#888; margin-left:6px;">ID: {{ p.user_id }}</span>
          </div>

          <span
            style="
              background:#ecfdf5;
              color:#15803d;
              padding:4px 12px;
              border-radius:20px;
              font-size:12px;
              font-weight:600;
            "
          >
            WALLET
          </span>
        </div>

        <!-- Cart -->
        <div v-if="p.cart && p.cart.length" style="margin-top: 14px;">
          <table style="width:100%; border-collapse: collapse;">
            <tr
              v-for="(item, index) in p.cart"
              :key="index"
              style="border-bottom: 1px solid #eee;"
            >
              <td style="padding:6px 0;">{{ item.name }}</td>
              <td style="width:60px;">x{{ item.quantity }}</td>
              <td style="width:90px; text-align:right;">₱{{ item.price }}</td>
            </tr>
          </table>

          <div
            style="
              margin-top: 10px;
              display:flex;
              justify-content:space-between;
              font-weight:bold;
              font-size:15px;
            "
          >
            <span>Total</span>
            <span>₱{{ getCartTotal(p.cart) }}</span>
          </div>
        </div>

        <!-- Actions -->
        <div style="margin-top: 16px; display:flex; gap:10px;">
          <button
            @click="handleApproveWallet(p.id)"
            style="
              background:#16a34a;
              color:white;
              border:none;
              padding:9px 16px;
              border-radius:8px;
              cursor:pointer;
              font-weight:600;
            "
          >
            Approve
          </button>

          <button
            @click="handleCancelWallet(p.id)"
            style="
              background:#fff;
              color:#dc2626;
              border:1px solid #dc2626;
              padding:9px 16px;
              border-radius:8px;
              cursor:pointer;
              font-weight:600;
            "
          >
            Reject
          </button>
        </div>
      </div>
    </div>

  </div>
</template>

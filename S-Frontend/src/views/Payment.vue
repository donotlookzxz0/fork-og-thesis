<script setup>
import { onMounted, ref } from "vue";
import { useCashPayment } from "../composables/useCashPayment";
import { useWalletPayment } from "../composables/useWalletPayment";
import { CashAPI } from "../services/cashApi";
import { WalletAPI } from "../services/walletApi";

// PrimeVue
import Card from "primevue/card";
import Button from "primevue/button";
import Divider from "primevue/divider";
import Tag from "primevue/tag";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

/* -----------------------
   TOAST
----------------------- */
const toast = useToast();

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

/* -----------------------
   LOADING STATE
----------------------- */
const refreshing = ref(false);

onMounted(() => {
  fetchCashPending();
  fetchWalletPending();
});

/* -----------------------
   🔄 REFRESH ALL PAYMENTS
----------------------- */
const refreshPayments = async () => {
  if (refreshing.value) return;

  refreshing.value = true;

  try {
    await Promise.all([
      fetchCashPending(),
      fetchWalletPending(),
    ]);

    // ✅ SUCCESS POPUP
    toast.add({
      severity: "success",
      summary: "Refreshed",
      detail: "Payments refreshed successfully",
      life: 2000,
    });

  } catch (err) {
    console.error("Refresh failed:", err);

    // ❌ ERROR POPUP
    toast.add({
      severity: "error",
      summary: "Error",
      detail: "Failed to refresh payments",
      life: 3000,
    });
  } finally {
    refreshing.value = false;
  }
};

/* -----------------------
   GENERATE CASH CODE
----------------------- */
const handleGenerate = async (id) => {
  try {
    const data = await generateCode(id);

    toast.add({
      severity: "success",
      summary: "Code Generated",
      detail: `Cash code: ${data.code}`,
      life: 2500,
    });

    fetchCashPending();
  } catch (err) {
    toast.add({
      severity: "error",
      summary: "Failed",
      detail: err?.message || "Failed to generate code",
      life: 3000,
    });
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

    toast.add({
      severity: "success",
      summary: "Approved",
      detail: "Wallet payment approved",
      life: 2000,
    });

    fetchWalletPending();
  } catch (err) {
    console.error(err);

    toast.add({
      severity: "error",
      summary: "Failed",
      detail: err?.response?.data?.error || "Wallet approval failed",
      life: 3000,
    });
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

    toast.add({
      severity: "success",
      summary: "Rejected",
      detail: "Wallet payment rejected",
      life: 2000,
    });

    fetchWalletPending();
  } catch (err) {
    console.error(err);

    toast.add({
      severity: "error",
      summary: "Failed",
      detail: err?.response?.data?.error || "Wallet cancel failed",
      life: 3000,
    });
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

  const reason = prompt("Reason for cancelling this cash request? (optional)");

  try {
    await CashAPI.cancel(id, reason);

    toast.add({
      severity: "success",
      summary: "Cancelled",
      detail: "Cash payment cancelled",
      life: 2000,
    });

    fetchCashPending();
  } catch (err) {
    console.error(err);

    toast.add({
      severity: "error",
      summary: "Failed",
      detail: err?.response?.data?.error || "Cancel failed",
      life: 3000,
    });
  }
};
</script>

<template>
  <div class="payment-wrapper">

    <!-- 🔔 TOAST POPUPS -->
    <Toast position="top-center" />

    <div class="payment">

      <!-- 🔥 TITLE + REFRESH -->
      <div class="title-row">
        <h1 class="title">Payment Approval Mode</h1>

        <Button
          icon="pi pi-refresh"
          label="Refresh"
          outlined
          severity="secondary"
          :loading="refreshing"
          @click="refreshPayments"
        />
      </div>

      <!-- =======================
           CASH PAYMENTS
      ======================= -->
      <Card class="section-card">
        <template #title>Cash Payments</template>

        <template #content>
          <div v-if="!cashPending.length" class="empty">
            No pending cash payments
          </div>

          <div
            v-for="p in cashPending"
            :key="`cash-${p.id}`"
            class="request-card"
          >
            <div class="header">
              <div>
                <strong>{{ p.username }}</strong>
                <span class="muted"> (ID: {{ p.user_id }})</span>
              </div>

              <Tag severity="warning" value="CASH" />
            </div>

            <!-- CART -->
            <div v-if="p.cart && p.cart.length" class="cart">
              <div
                v-for="(item, index) in p.cart"
                :key="index"
                class="row"
              >
                <span>{{ item.name }}</span>
                <span>x{{ item.quantity }}</span>
                <span>₱{{ item.price }}</span>
              </div>

              <div class="total">
                Total: ₱{{ getCartTotal(p.cart) }}
              </div>
            </div>

            <!-- CODE -->
            <div v-if="p.code" class="code">
              Cash Code: <span>{{ p.code }}</span>
            </div>

            <!-- ACTIONS -->
            <div class="actions">
              <Button
                label="Generate Code"
                icon="pi pi-key"
                :disabled="!!p.code"
                @click="handleGenerate(p.id)"
              />

              <Button
                label="Cancel"
                icon="pi pi-times"
                severity="danger"
                text
                @click="cancelRequest(p.id)"
              />
            </div>
          </div>
        </template>
      </Card>

      <Divider />

      <!-- =======================
           WALLET PAYMENTS
      ======================= -->
      <Card class="section-card">
        <template #title>Wallet Payments</template>

        <template #content>
          <div v-if="!walletPending.length" class="empty">
            No pending wallet payments
          </div>

          <div
            v-for="p in walletPending"
            :key="`wallet-${p.id}`"
            class="request-card"
          >
            <div class="header">
              <div>
                <strong>{{ p.username }}</strong>
                <span class="muted"> (ID: {{ p.user_id }})</span>
              </div>

              <Tag severity="success" value="WALLET" />
            </div>

            <!-- CART -->
            <div v-if="p.cart && p.cart.length" class="cart">
              <div
                v-for="(item, index) in p.cart"
                :key="index"
                class="row"
              >
                <span>{{ item.name }}</span>
                <span>x{{ item.quantity }}</span>
                <span>₱{{ item.price }}</span>
              </div>

              <div class="total">
                Total: ₱{{ getCartTotal(p.cart) }}
              </div>
            </div>

            <!-- ACTIONS -->
            <div class="actions">
              <Button
                label="Approve"
                icon="pi pi-check"
                severity="success"
                @click="handleApproveWallet(p.id)"
              />

              <Button
                label="Reject"
                icon="pi pi-times"
                severity="danger"
                text
                @click="handleCancelWallet(p.id)"
              />
            </div>
          </div>
        </template>
      </Card>

    </div>
  </div>
</template>

<style scoped>
/* CENTER PAGE */
.payment-wrapper {
  min-height: calc(100vh - 40px);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 40px;
}

.payment {
  width: 100%;
  max-width: 900px;
  padding: 20px;
}

/* 🔥 TITLE ROW WITH REFRESH */
.title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

/* HEADER */
.title {
  color: #ffffff;
  font-size: 1.8rem;
}

/* SECTION CARD */
.section-card {
  background: #1f1f1f;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
}

/* EMPTY STATE */
.empty {
  text-align: center;
  color: #9ca3af;
  padding: 40px 0;
  font-size: 1rem;
}

/* REQUEST CARD */
.request-card {
  background: #242424;
  border-radius: 14px;
  padding: 16px;
  margin-bottom: 16px;
  border: 1px solid rgba(255, 255, 255, 0.06);
}

/* HEADER ROW */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.muted {
  color: #9ca3af;
}

/* CART */
.cart {
  margin-bottom: 12px;
}

.row {
  display: grid;
  grid-template-columns: 1fr auto auto;
  gap: 10px;
  padding: 4px 0;
  color: #e5e7eb;
}

.total {
  margin-top: 8px;
  font-weight: bold;
  color: #34d399;
}

/* CODE */
.code {
  margin-bottom: 12px;
  font-size: 1.1rem;
  color: #fbbf24;
}

.code span {
  font-weight: bold;
  letter-spacing: 2px;
}

/* ACTIONS */
.actions {
  display: flex;
  gap: 10px;
  justify-content: flex-end;
}
</style>

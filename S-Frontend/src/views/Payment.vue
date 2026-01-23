<script setup>
import { onMounted, ref } from "vue";
import { useCashPayment } from "../composables/useCashPayment";
import { useWalletPayment } from "../composables/useWalletPayment";
import { CashAPI } from "../services/cashApi";
import { WalletAPI } from "../services/walletApi";

import Card from "primevue/card";
import Button from "primevue/button";
import DataTable from "primevue/datatable";
import Column from "primevue/column";
import Tag from "primevue/tag";
import Divider from "primevue/divider";
import ConfirmDialog from "primevue/confirmdialog";
import { useConfirm } from "primevue/useconfirm";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

const confirm = useConfirm();
const toast = useToast();

/* CASH */
const { pending: cashPending, fetchPending: fetchCashPending, generateCode } =
  useCashPayment();

/* WALLET */
const {
  pending: walletPending,
  fetchPending: fetchWalletPending,
  approve,
} = useWalletPayment();

onMounted(() => {
  fetchCashPending();
  fetchWalletPending();
});

const getCartTotal = (cart = []) =>
  cart.reduce((sum, item) => sum + item.price * item.quantity, 0);

/* CASH CODE */
const handleGenerate = async (id) => {
  try {
    const data = await generateCode(id);
    toast.add({
      severity: "success",
      summary: "Code Generated",
      detail: `Cash Code: ${data.code}`,
      life: 3000,
    });
    fetchCashPending();
  } catch (err) {
    toast.add({
      severity: "error",
      summary: "Error",
      detail: err?.message || "Failed to generate code",
    });
  }
};

/* CANCEL CASH */
const cancelRequest = (id) => {
  confirm.require({
    message: "Cancel this cash payment request?",
    header: "Confirm Cancel",
    icon: "pi pi-exclamation-triangle",
    accept: async () => {
      try {
        await CashAPI.cancel(id);
        toast.add({
          severity: "info",
          summary: "Cancelled",
          detail: "Cash payment cancelled",
        });
        fetchCashPending();
      } catch (err) {
        toast.add({
          severity: "error",
          summary: "Error",
          detail: err?.response?.data?.error || "Cancel failed",
        });
      }
    },
  });
};

/* APPROVE WALLET */
const handleApproveWallet = (id) => {
  confirm.require({
    message: "Approve this wallet payment?",
    header: "Approve Payment",
    icon: "pi pi-check-circle",
    accept: async () => {
      try {
        await approve(id);
        toast.add({
          severity: "success",
          summary: "Approved",
          detail: "Wallet payment approved",
        });
        fetchWalletPending();
      } catch (err) {
        toast.add({
          severity: "error",
          summary: "Error",
          detail: err?.response?.data?.error || "Wallet approval failed",
        });
      }
    },
  });
};

/* REJECT WALLET */
const handleCancelWallet = (id) => {
  confirm.require({
    message: "Reject this wallet payment?",
    header: "Reject Payment",
    icon: "pi pi-times-circle",
    accept: async () => {
      try {
        await WalletAPI.cancel(id);
        toast.add({
          severity: "warn",
          summary: "Rejected",
          detail: "Wallet payment rejected",
        });
        fetchWalletPending();
      } catch (err) {
        toast.add({
          severity: "error",
          summary: "Error",
          detail: err?.response?.data?.error || "Wallet cancel failed",
        });
      }
    },
  });
};
</script>

<template>
  <Toast />
  <ConfirmDialog />

  <div class="grid p-4 gap-4">
    <!-- ================= CASH PAYMENTS ================= -->
    <div class="col-12 md:col-6">
      <h2 class="mb-3">💵 Cash Payments</h2>

      <Card
        v-for="p in cashPending"
        :key="`cash-${p.id}`"
        class="mb-4 shadow-2"
      >
        <template #title>
          {{ p.username }}
          <small class="ml-2 text-500">ID: {{ p.user_id }}</small>
        </template>

        <template #content>
          <!-- Cart -->
          <DataTable
            v-if="p.cart?.length"
            :value="p.cart"
            size="small"
            stripedRows
            class="mb-3"
          >
            <Column field="name" header="Item" />
            <Column field="quantity" header="Qty" style="width:80px" />
            <Column field="price" header="Price" style="width:100px" />
          </DataTable>

          <div class="flex justify-content-between mb-2 font-bold">
            <span>Total</span>
            <span>₱{{ getCartTotal(p.cart) }}</span>
          </div>

          <!-- Code -->
          <div v-if="p.code" class="mb-3">
            <Tag severity="success" icon="pi pi-key" value="Code Generated" />
            <div class="mt-2 text-xl font-bold text-primary">
              {{ p.code }}
            </div>
          </div>

          <!-- Actions -->
          <div class="flex gap-2">
            <Button
              label="Generate Code"
              icon="pi pi-qrcode"
              size="small"
              :disabled="!!p.code"
              @click="handleGenerate(p.id)"
            />

            <Button
              label="Cancel"
              icon="pi pi-times"
              severity="danger"
              size="small"
              outlined
              @click="cancelRequest(p.id)"
            />
          </div>
        </template>
      </Card>
    </div>

    <!-- ================= WALLET PAYMENTS ================= -->
    <div class="col-12 md:col-6">
      <h2 class="mb-3">👛 Wallet Payments</h2>

      <Card
        v-for="p in walletPending"
        :key="`wallet-${p.id}`"
        class="mb-4 shadow-2"
      >
        <template #title>
          {{ p.username }}
          <small class="ml-2 text-500">ID: {{ p.user_id }}</small>
        </template>

        <template #content>
          <DataTable
            v-if="p.cart?.length"
            :value="p.cart"
            size="small"
            stripedRows
            class="mb-3"
          >
            <Column field="name" header="Item" />
            <Column field="quantity" header="Qty" style="width:80px" />
            <Column field="price" header="Price" style="width:100px" />
          </DataTable>

          <div class="flex justify-content-between mb-3 font-bold">
            <span>Total</span>
            <span>₱{{ getCartTotal(p.cart) }}</span>
          </div>

          <div class="flex gap-2">
            <Button
              label="Approve"
              icon="pi pi-check"
              severity="success"
              size="small"
              @click="handleApproveWallet(p.id)"
            />

            <Button
              label="Reject"
              icon="pi pi-times"
              severity="danger"
              size="small"
              outlined
              @click="handleCancelWallet(p.id)"
            />
          </div>
        </template>
      </Card>
    </div>
  </div>
</template>

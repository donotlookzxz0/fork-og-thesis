import { ref } from "vue";
import { WalletAPI } from "../services/walletApi";

export function useWalletPayment() {
  const pending = ref([]);

  /* -----------------------
     FETCH PENDING WALLET PAYMENTS (ADMIN)
  ----------------------- */
  const fetchPending = async () => {
    try {
      const res = await WalletAPI.getPending();
      pending.value = res.data;
    } catch (err) {
      console.error("Failed to fetch pending wallet payments", err);
      pending.value = [];
      throw err;
    }
  };

  /* -----------------------
     APPROVE WALLET PAYMENT (ADMIN)
  ----------------------- */
  const approve = async (id) => {
    try {
      const res = await WalletAPI.approve(id);
      return res.data;
    } catch (err) {
      console.error("Failed to approve wallet payment", err);
      throw err;
    }
  };

  return {
    pending,
    fetchPending,
    approve,
  };
}

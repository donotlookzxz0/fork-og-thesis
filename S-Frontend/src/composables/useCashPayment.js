import { ref } from "vue";
import { CashAPI } from "../services/cashApi";

export function useCashPayment() {
  const pending = ref([]);

  /* -----------------------
     FETCH PENDING CASH PAYMENTS (ADMIN)
  ----------------------- */
  const fetchPending = async () => {
    try {
      const res = await CashAPI.getPending();
      pending.value = res.data;
    } catch (err) {
      console.error("Failed to fetch pending cash payments", err);
      pending.value = [];
      throw err;
    }
  };

  /* -----------------------
     GENERATE CASH CODE (ADMIN)
  ----------------------- */
  const generateCode = async (id) => {
    try {
      const res = await CashAPI.generateCode(id);
      return res.data;
    } catch (err) {
      console.error("Failed to generate cash code", err);
      throw err;
    }
  };

  return { pending, fetchPending, generateCode };
}

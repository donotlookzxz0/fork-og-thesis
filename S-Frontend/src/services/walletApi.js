import api from "./api";

export const WalletAPI = {
  // 🔍 Fetch pending wallet payments (ADMIN)
  getPending: () =>
    api.get("/payment/admin/wallet/pending"),

  // ✅ Approve wallet payment (ADMIN)
  approve: (id) =>
    api.post(`/payment/admin/wallet/approve/${id}`),

  // ❌ Cancel / Reject wallet payment (ADMIN) — NEW
  cancel: (id) =>
    api.post(`/payment/admin/wallet/cancel/${id}`),
};

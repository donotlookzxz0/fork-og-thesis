import api from "./api";

export const CashAPI = {
  getPending: () => api.get("/payment/admin/cash/pending"),

  generateCode: (id) =>
    api.post(`/payment/admin/cash/generate-code/${id}`),

  cancel: (id, reason) =>
    api.post(`/payment/admin/cash/cancel/${id}`, { reason }),
};

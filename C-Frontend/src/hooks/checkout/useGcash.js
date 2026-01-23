import api from "../../api/axios";

export function useGcash({ cart, totalPrice }) {
  const payWithGcash = async () => {
    const intentRes = await api.post("/payment/intent", {
      amount: totalPrice * 100,
      currency: "PHP",
    });

    const checkoutRes = await api.post("/payment/checkout", {
      payment_intent_id: intentRes.data.id,
      success_url: `${window.location.origin}/success`,
      cancel_url: `${window.location.origin}/cancel`,
      cart,
    });

    window.location.href = checkoutRes.data.checkoutUrl;
  };

  return { payWithGcash };
}
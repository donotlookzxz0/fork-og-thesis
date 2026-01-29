import { defineStore } from "pinia"
import { ref, computed, watch } from "vue"

export const useCartStore = defineStore("cart", () => {

  /* ---------------- STATE ---------------- */
  const cart = ref(JSON.parse(localStorage.getItem("pos_cart") || "[]"))

  /* ---------------- TOTAL ---------------- */
  const total = computed(() =>
    cart.value.reduce((sum, i) => sum + i.price * i.quantity, 0)
  )

  /* ---------------- ADD ITEM ---------------- */
  const addItem = (item) => {
    const existing = cart.value.find(c => c.item_id === item.id)

    if (existing) {
      // ✅ immutable update so computed total always refreshes
      cart.value = cart.value.map(i =>
        i.item_id === item.id
          ? { ...i, quantity: i.quantity + 1 }
          : i
      )
    } else {
      cart.value = [
        ...cart.value,
        {
          item_id: item.id,
          name: item.name,
          price: item.price,
          quantity: 1
        }
      ]
    }
  }

  /* ---------------- QTY ---------------- */
  const increaseQty = (row) => {
    cart.value = cart.value.map(i =>
      i.item_id === row.item_id
        ? { ...i, quantity: i.quantity + 1 }
        : i
    )
  }

  const decreaseQty = (row) => {
    cart.value = cart.value.map(i =>
      i.item_id === row.item_id
        ? { ...i, quantity: Math.max(1, i.quantity - 1) }
        : i
    )
  }

  /* ---------------- REMOVE ---------------- */
  const removeItem = (row) => {
    cart.value = cart.value.filter(i => i.item_id !== row.item_id)
  }

  /* ---------------- CLEAR ---------------- */
  const clearCart = () => {
    cart.value = []
  }

  /* ---------------- PERSIST ---------------- */
  watch(cart, (val) => {
    localStorage.setItem("pos_cart", JSON.stringify(val))
  }, { deep: true })

  return {
    cart,
    total,
    addItem,
    increaseQty,
    decreaseQty,
    removeItem,
    clearCart
  }
})

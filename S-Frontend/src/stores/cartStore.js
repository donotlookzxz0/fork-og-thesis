import { defineStore } from "pinia"
import { ref, computed, watch } from "vue"

export const useCartStore = defineStore("cart", () => {

  const cart = ref(JSON.parse(localStorage.getItem("pos_cart") || "[]"))

  const total = computed(() =>
    cart.value.reduce(
      (sum, i) => sum + Number(i.price) * Number(i.quantity),
      0
    )
  )

  const addItem = (item) => {
    const existing = cart.value.find(c => c.item_id === item.id)

    if (existing) {
      existing.quantity++
    } else {
      cart.value.push({
        item_id: item.id,
        name: item.name,
        price: Number(item.price),
        quantity: 1
      })
    }
  }

  const increaseQty = (row) => {
    row.quantity++
  }

  const decreaseQty = (row) => {
    if (row.quantity > 1) row.quantity--
  }

  const removeItem = (row) => {
    cart.value = cart.value.filter(i => i.item_id !== row.item_id)
  }

  const clearCart = () => {
    cart.value = []
  }

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

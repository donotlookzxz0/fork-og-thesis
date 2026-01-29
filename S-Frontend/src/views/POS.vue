<script setup>
import { ref, onMounted, onBeforeUnmount } from "vue"
import api from "../services/api"
import { useCartStore } from "../stores/cartStore"

import InputText from "primevue/inputtext"
import Button from "primevue/button"
import DataTable from "primevue/datatable"
import Column from "primevue/column"
import Card from "primevue/card"
import Divider from "primevue/divider"
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()
const cartStore = useCartStore()

const manualBarcode = ref("")
const nameQuery = ref("")
const nameResults = ref([])

const userId = ref(null)
const loadingUser = ref(true)
const checkingOut = ref(false)

let scanBuffer = ""
let scanTimeout = null
let nameSearchTimeout = null

const getItemByBarcode = (barcode) =>
  api.get(`/items/barcode/${barcode}`)

const searchItemsByName = (q) =>
  api.get(`/items/?search=${encodeURIComponent(q)}`)

const createTransaction = (payload) =>
  api.post("/sales", payload)

const fetchMe = async () => {
  try {
    const res = await api.get("/users/me")
    userId.value = res.data.id
  } catch {
    toast.add({
      severity: "error",
      summary: "Session Expired",
      detail: "Please login again",
      life: 3000,
    })
  } finally {
    loadingUser.value = false
  }
}

const handleKeydown = async (e) => {
  if (["Shift", "Alt", "Control"].includes(e.key)) return

  if (e.key === "Enter") {
    if (scanBuffer.length > 0) {
      await addByBarcode(scanBuffer)
      scanBuffer = ""
    }
    return
  }

  scanBuffer += e.key

  clearTimeout(scanTimeout)
  scanTimeout = setTimeout(() => {
    scanBuffer = ""
  }, 300)
}

onMounted(() => {
  fetchMe()
  window.addEventListener("keydown", handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown)
})

const addByBarcode = async (barcode) => {
  try {
    const res = await getItemByBarcode(barcode)
    cartStore.addItem(res.data)
  } catch {
    toast.add({
      severity: "warn",
      summary: "Not Found",
      detail: "Item not found",
      life: 2000,
    })
  }
}

const addManualBarcode = async () => {
  if (!manualBarcode.value) return
  await addByBarcode(manualBarcode.value)
  manualBarcode.value = ""
}

const searchByName = () => {
  clearTimeout(nameSearchTimeout)
  nameSearchTimeout = setTimeout(async () => {
    if (!nameQuery.value) {
      nameResults.value = []
      return
    }
    try {
      const res = await searchItemsByName(nameQuery.value)
      nameResults.value = res.data.slice(0, 6)
    } catch {
      nameResults.value = []
    }
  }, 250)
}

const addFirstNameMatch = () => {
  if (nameResults.value.length) {
    addFromName(nameResults.value[0])
  }
}

const addFromName = (item) => {
  cartStore.addItem(item)
  nameQuery.value = ""
  nameResults.value = []
}

const addManualName = async () => {
  if (!nameQuery.value) return
  try {
    const res = await searchItemsByName(nameQuery.value)
    if (res.data.length) {
      cartStore.addItem(res.data[0])
      nameQuery.value = ""
      nameResults.value = []
    } else {
      toast.add({
        severity: "warn",
        summary: "Not Found",
        detail: "No matching item",
        life: 2000,
      })
    }
  } catch {
    toast.add({
      severity: "error",
      summary: "Search Failed",
      detail: "Could not search items",
      life: 2000,
    })
  }
}

const increaseQty = cartStore.increaseQty
const decreaseQty = cartStore.decreaseQty
const removeItem = cartStore.removeItem
const total = cartStore.total

const checkout = async () => {
  if (checkingOut.value) return

  if (loadingUser.value) {
    toast.add({
      severity: "info",
      summary: "Loading",
      detail: "Loading user...",
      life: 2000,
    })
    return
  }

  if (!userId.value) {
    toast.add({
      severity: "error",
      summary: "Auth Error",
      detail: "Not authenticated. Please login again.",
      life: 3000,
    })
    return
  }

  if (!cartStore.cart.length) {
    toast.add({
      severity: "warn",
      summary: "Empty Cart",
      detail: "Cart is empty",
      life: 2000,
    })
    return
  }

  checkingOut.value = true

  try {
    const payload = {
      user_id: userId.value,
      items: cartStore.cart.map(i => ({
        item_id: i.item_id,
        quantity: i.quantity
      }))
    }

    await createTransaction(payload)
    cartStore.clearCart()

    toast.add({
      severity: "success",
      summary: "Success",
      detail: "Transaction completed successfully",
      life: 2500,
    })
  } catch (err) {
    toast.add({
      severity: "error",
      summary: "Checkout Failed",
      detail: err.response?.data?.error || "Checkout failed",
      life: 3000,
    })
  } finally {
    checkingOut.value = false
  }
}
</script>

<template>
  <div class="pos-wrapper">
    <Toast position="top-center" />

    <div class="pos">
      <h1 class="title">POS Mode</h1>

      <div class="scan-row">
        <InputText
          v-model="manualBarcode"
          placeholder="Scan or type barcode"
          @keyup.enter="addManualBarcode"
        />
        <Button label="Add" icon="pi pi-plus" @click="addManualBarcode" />
      </div>

      <div class="scan-row">
        <InputText
          v-model="nameQuery"
          placeholder="Search by item name"
          @input="searchByName"
          @keyup.enter="addFirstNameMatch"
        />
        <Button label="Add" icon="pi pi-plus" @click="addManualName" />
      </div>

      <div v-if="nameResults.length" class="results">
        <div
          v-for="item in nameResults"
          :key="item.id"
          class="result-row"
          @click="addFromName(item)"
        >
          {{ item.name }} — ₱{{ item.price }}
        </div>
      </div>

      <small class="hint">
        Scan barcode or search by name
      </small>

      <Card class="cart-card">
        <template #title>Current Cart</template>

        <template #content>
          <DataTable
            :value="cartStore.cart"
            v-if="cartStore.cart.length"
            responsiveLayout="scroll"
          >
            <Column field="name" header="Item" />

            <Column header="Qty">
              <template #body="{ data }">
                <Button icon="pi pi-minus" text @click="decreaseQty(data)" />
                <span class="qty">{{ data.quantity }}</span>
                <Button icon="pi pi-plus" text @click="increaseQty(data)" />
              </template>
            </Column>

            <Column header="Price">
              <template #body="{ data }">
                ₱{{ data.price }}
              </template>
            </Column>

            <Column header="Total">
              <template #body="{ data }">
                ₱{{ (data.price * data.quantity).toFixed(2) }}
              </template>
            </Column>

            <Column header="">
              <template #body="{ data }">
                <Button
                  icon="pi pi-trash"
                  severity="danger"
                  text
                  @click="removeItem(data)"
                />
              </template>
            </Column>
          </DataTable>

          <div v-else class="empty">
            Scan or search an item to start
          </div>

          <Divider />

          <div class="total-row">
            <h2>Total</h2>
            <h2 class="amount">₱{{ total.toFixed(2) }}</h2>
          </div>

          <Button
            label="PAY"
            icon="pi pi-credit-card"
            class="pay"
            :disabled="!cartStore.cart.length || loadingUser || checkingOut"
            :loading="checkingOut"
            @click="checkout"
          />
        </template>
      </Card>
    </div>
  </div>
</template>

<style scoped>
.pos-wrapper {
  min-height: calc(100vh - 40px);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 40px;
}

.pos {
  width: 100%;
  max-width: 720px;
  padding: 20px;
}

.title {
  color: #ffffff;
  font-size: 1.8rem;
  margin-bottom: 16px;
}

.scan-row {
  display: flex;
  gap: 10px;
  margin-bottom: 8px;
}

.results {
  background: #2a2a2a;
  border-radius: 12px;
  margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,0.08);
}

.result-row {
  padding: 10px 14px;
  cursor: pointer;
}

.result-row:hover {
  background: #3a3a3a;
}

.hint {
  color: #9ca3af;
  display: block;
  margin-bottom: 16px;
}

.cart-card {
  background: #1f1f1f;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
}

.empty {
  text-align: center;
  color: #9ca3af;
  padding: 40px 0;
  font-size: 1rem;
}

.qty {
  margin: 0 8px;
  font-weight: bold;
}

.total-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.amount {
  color: #34d399;
  font-weight: bold;
}

.pay {
  width: 100%;
  height: 54px;
  font-size: 1.2rem;
}
</style>

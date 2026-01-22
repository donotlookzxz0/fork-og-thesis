<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from "vue"
import api from "../services/api"

// PrimeVue components
import InputText from "primevue/inputtext"
import Button from "primevue/button"
import DataTable from "primevue/datatable"
import Column from "primevue/column"
import Message from "primevue/message"

/* ---------------- STATE ---------------- */
const manualBarcode = ref("")
const cart = ref([])
const message = ref("")
const userId = ref(null)   // ✅ added

let scanBuffer = ""
let scanTimeout = null

/* ---------------- API ---------------- */
const getItemByBarcode = (barcode) =>
  api.get(`items/barcode/${barcode}`)

const createTransaction = (payload) =>
  api.post("sales", payload)

/* ---------------- AUTH USER ---------------- */
const fetchMe = async () => {
  const res = await api.get("users/me")
  userId.value = res.data.id
}

/* ---------------- SCANNER (AUTO) ---------------- */
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
  fetchMe() // ✅ added
  window.addEventListener("keydown", handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown)
})

/* ---------------- CART ---------------- */
const addByBarcode = async (barcode) => {
  try {
    const res = await getItemByBarcode(barcode)
    const item = res.data

    const existing = cart.value.find(c => c.item_id === item.id)

    if (existing) {
      existing.quantity++
    } else {
      cart.value.push({
        item_id: item.id,
        name: item.name,
        price: item.price,
        quantity: 1
      })
    }
  } catch {
    message.value = "Item not found"
  }
}

const addManual = async () => {
  if (!manualBarcode.value) return
  await addByBarcode(manualBarcode.value)
  manualBarcode.value = ""
}

const increaseQty = (row) => row.quantity++
const decreaseQty = (row) => {
  if (row.quantity > 1) row.quantity--
}

const removeItem = (row) => {
  cart.value = cart.value.filter(i => i.item_id !== row.item_id)
}

/* ---------------- TOTAL ---------------- */
const total = computed(() =>
  cart.value.reduce((sum, i) => sum + i.price * i.quantity, 0)
)

/* ---------------- CHECKOUT ---------------- */
const checkout = async () => {
  try {
    const payload = {
      user_id: userId.value,   // ✅ required by backend
      items: cart.value.map(i => ({
        item_id: i.item_id,
        quantity: i.quantity
      }))
    }

    await createTransaction(payload)

    cart.value = []
    message.value = "Transaction completed"
  } catch {
    message.value = "Checkout failed"
  }
}
</script>

<template>
  <div class="pos">
    <h1>POS System</h1>

    <div class="scan-row">
      <InputText
        v-model="manualBarcode"
        placeholder="Type barcode (optional)"
        @keyup.enter="addManual"
      />
      <Button label="Add" icon="pi pi-plus" @click="addManual" />
    </div>

    <small class="hint">
      Scanner ready. You can scan anytime without focusing an input.
    </small>

    <DataTable
      :value="cart"
      v-if="cart.length"
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

    <h2>Total: ₱{{ total.toFixed(2) }}</h2>

    <Button
      label="PAY"
      icon="pi pi-credit-card"
      class="pay"
      :disabled="!cart.length"
      @click="checkout"
    />

    <Message v-if="message" severity="info" class="mt-3">
      {{ message }}
    </Message>
  </div>
</template>

<style scoped>
.pos {
  padding: 20px;
}

.scan-row {
  display: flex;
  gap: 10px;
  margin-bottom: 10px;
}

.hint {
  color: #666;
  display: block;
  margin-bottom: 12px;
}

.qty {
  margin: 0 8px;
  font-weight: bold;
}

.pay {
  margin-top: 16px;
  height: 50px;
  font-size: 1.1rem;
}
</style>

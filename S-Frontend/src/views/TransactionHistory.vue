<script setup>
import { ref, onMounted, watch, computed } from "vue"
import { useRouter } from "vue-router"
import api from "../services/api"

/* ---------------- STATE ---------------- */
const router = useRouter()

const transactions = ref([])
const allTransactions = ref([])

const loading = ref(false)
const error = ref("")

const page = ref(1)
const perPage = ref(10)

const userId = ref(null)
const authChecked = ref(false)

/* ---------------- AUTH USER ---------------- */
const fetchMe = async () => {
  try {
    const res = await api.get("/users/me")
    userId.value = res.data.id
    authChecked.value = true
  } catch (err) {
    console.error("Auth error:", err)
    authChecked.value = true
    error.value = "Session expired. Please login again."

    setTimeout(() => {
      router.push("/login")
    }, 800)

    throw err
  }
}

/* ---------------- PAGINATION ---------------- */
const totalPages = computed(() => {
  return Math.max(1, Math.ceil(allTransactions.value.length / perPage.value))
})

const applyPagination = () => {
  if (page.value > totalPages.value) page.value = totalPages.value
  if (page.value < 1) page.value = 1

  const start = (page.value - 1) * perPage.value
  const end = start + perPage.value
  transactions.value = allTransactions.value.slice(start, end)
}

const nextPage = () => {
  if (page.value < totalPages.value) page.value++
}

const prevPage = () => {
  if (page.value > 1) page.value--
}

watch(page, applyPagination)
watch(perPage, () => {
  page.value = 1
  applyPagination()
})

/* ---------------- TOTAL HELPER ---------------- */
const getTransactionTotal = (items = []) => {
  return items.reduce(
    (sum, item) => sum + item.price_at_sale * item.quantity,
    0
  )
}

/* ---------------- FETCH TRANSACTIONS ---------------- */
const fetchTransactions = async () => {
  loading.value = true
  error.value = ""

  try {
    const res = await api.get("/sales")

    allTransactions.value = res.data.sort(
      (a, b) => new Date(b.date) - new Date(a.date)
    )

    page.value = 1
    applyPagination()

  } catch (err) {
    console.error("Transactions error:", err)

    if (err.response?.status === 401) {
      error.value = "Session expired. Please login again."
      setTimeout(() => {
        router.push("/login")
      }, 800)
    } else {
      error.value = err.response?.data?.error || "Failed to load transactions"
    }

  } finally {
    loading.value = false
  }
}

/* ---------------- INIT ---------------- */
onMounted(async () => {
  try {
    await fetchMe()
    await fetchTransactions()
  } catch {}
})
</script>

<template>
  <div class="transactions">

    <!-- 🔥 TITLE AT VERY TOP (FIXED POSITION) -->
    <h1 class="title">
      <i class="pi pi-receipt"></i>
      Transaction History
    </h1>

    <!-- CONTROLS UNDER TITLE -->
    <div class="top-controls">
      <button @click="prevPage" :disabled="page === 1">◀ Prev</button>

      <span class="page-info">
        Page {{ page }} of {{ totalPages }}
      </span>

      <button @click="nextPage" :disabled="page === totalPages">
        Next ▶
      </button>

      <div class="per-page">
        <label>Records per page</label>
        <input
          type="number"
          min="1"
          max="100"
          v-model.number="perPage"
        />
      </div>
    </div>

    <!-- 🔄 Loading -->
    <p v-if="loading">Loading transactions...</p>

    <!-- ❌ Error -->
    <p v-if="error" class="error">{{ error }}</p>

    <!-- 🟢 Table -->
    <table v-if="!loading && transactions.length">
      <thead>
        <tr>
          <th>ID</th>
          <th>Date</th>
          <th>User</th>
          <th>Items</th>
          <th>Total</th>
        </tr>
      </thead>

      <tbody>
        <tr v-for="t in transactions" :key="t.transaction_id">
          <td>{{ t.transaction_id }}</td>
          <td>{{ new Date(t.date).toLocaleString() }}</td>
          <td>{{ t.user_id }}</td>

          <!-- ITEMS -->
          <td>
            <ul>
              <li v-for="item in t.items" :key="item.item_id">
                {{ item.item_name }} ({{ item.category }})
                — Qty: {{ item.quantity }}
                — ₱{{ item.price_at_sale }}
              </li>
            </ul>
          </td>

          <!-- TOTAL -->
          <td style="font-weight: bold; color: #34d399;">
            ₱{{ getTransactionTotal(t.items).toFixed(2) }}
          </td>
        </tr>
      </tbody>
    </table>

    <!-- 🟡 Empty -->
    <p v-if="!loading && !transactions.length && !error">
      No transactions found
    </p>
  </div>
</template>

<style scoped>
.transactions {
  padding: 20px;
}

/* 🔥 SAME SIZE AS PAYMENT.VUE */
.title {
  color: #ffffff;
  font-size: 1.8rem;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.top-controls {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.top-controls button {
  height: 38px;
  padding: 0 14px;
  cursor: pointer;
}

.top-controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-info {
  font-weight: 600;
}

.per-page {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.per-page input {
  height: 38px;
  width: 120px;
  padding: 0 10px;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 16px;
}

th,
td {
  border: 1px solid #ddd;
  padding: 10px;
  vertical-align: top;
}

ul {
  padding-left: 16px;
  margin: 0;
}

.error {
  color: #ef4444;
  font-weight: 500;
}
</style>

<script setup>
import { ref, onMounted, watch, computed } from "vue"
import api from "../services/api"

const transactions = ref([])
const allTransactions = ref([])

const loading = ref(false)
const error = ref("")

const page = ref(1)
const perPage = ref(10)

const userId = ref(null) // ✅ added

/* ---------------- AUTH USER ---------------- */
const fetchMe = async () => {
  const res = await api.get("users/me")
  userId.value = res.data.id
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

/* ---------------- FETCH ---------------- */
const fetchTransactions = async () => {
  loading.value = true
  error.value = ""
  try {
    const res = await api.get("sales")

    // ✅ MOST RECENT FIRST
    allTransactions.value = res.data.sort(
      (a, b) => new Date(b.date) - new Date(a.date)
    )

    page.value = 1
    applyPagination()
  } catch (err) {
    error.value = err.response?.data?.error || "Failed to load transactions"
  } finally {
    loading.value = false
  }
}

const nextPage = () => {
  if (page.value < totalPages.value) page.value++
}

const prevPage = () => {
  if (page.value > 1) page.value--
}

watch(page, applyPagination)
watch(perPage, applyPagination)

/* ---------------- INIT ---------------- */
onMounted(async () => {
  await fetchMe()        // ✅ added
  fetchTransactions()
})
</script>

<template>
  <div class="transactions">
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

    <h1><i class="pi pi-receipt"></i> Transaction History</h1>

    <p v-if="loading">Loading...</p>
    <p v-if="error" class="error">{{ error }}</p>

    <table v-if="!loading && transactions.length">
      <thead>
        <tr>
          <th>ID</th>
          <th>Date</th>
          <th>User</th>
          <th>Items</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="t in transactions" :key="t.transaction_id">
          <td>{{ t.transaction_id }}</td>
          <td>{{ new Date(t.date).toLocaleString() }}</td>
          <td>{{ t.user_id }}</td>
          <td>
            <ul>
              <li v-for="item in t.items" :key="item.item_id">
                {{ item.item_name }} ({{ item.category }})
                — Qty: {{ item.quantity }}
                — {{ item.price_at_sale }}
              </li>
            </ul>
          </td>
        </tr>
      </tbody>
    </table>

    <p v-if="!loading && !transactions.length">
      No transactions found
    </p>
  </div>
</template>

<style scoped>
.transactions {
  padding: 20px;
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
}
</style>

<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

import Card from "primevue/card"
import DataTable from "primevue/datatable"
import Column from "primevue/column"
import ProgressSpinner from "primevue/progressspinner"

const toast = useToast()

console.log("✅ Metrics page script started")

const loading = ref(true)

const forecastRows = ref([])
const stockoutRows = ref([])
const stockoutGlobal = ref([])
const movementRows = ref([])
const movementGlobal = ref([])

/* ---------------- API CALLS ---------------- */

const getForecast = () => api.get("/metrics/forecast")
const getStockout = () => api.get("/metrics/stockout-risk")
const getMovement = () => api.get("/metrics/item-movement")

/* ---------------- PARSERS ---------------- */

const parseForecast = (metrics = {}) => {
  const rows = []

  Object.keys(metrics || {}).forEach(category => {
    const ranges = metrics[category] || {}

    Object.keys(ranges).forEach(range => {
      const m = ranges[range] || {}

      rows.push({
        category,
        range,
        mae: Number(m.mae ?? 0).toFixed(2),
        rmse: Number(m.rmse ?? 0).toFixed(2),
        mape: Number(m.mape ?? 0).toFixed(2) + "%"
      })
    })
  })

  return rows
}

const objToRows = (obj = {}) =>
  Object.keys(obj || {}).map(k => ({
    name: k,
    value: typeof obj[k] === "object"
      ? JSON.stringify(obj[k])
      : obj[k]
  }))

const parseCategory = (metrics = {}, maeKey) =>
  Object.keys(metrics || {}).map(cat => ({
    category: cat,
    accuracy: metrics[cat]?.accuracy ?? 0,
    macro_f1: metrics[cat]?.macro_f1 ?? 0,
    total_items: metrics[cat]?.total_items ?? 0,
    [maeKey]: metrics[cat]?.[maeKey] ?? 0
  }))

/* ---------------- LOAD DATA ---------------- */

const loadMetrics = async () => {
  console.log("🚀 loadMetrics() running")

  try {
    loading.value = true

    // 🔹 FORECAST
    const f = await getForecast()
    console.log("FORECAST RESPONSE:", f.data)

    if (f?.data?.success) {
      forecastRows.value = parseForecast(f.data.metrics)
    }

    // 🔹 STOCKOUT
    const s = await getStockout()
    console.log("STOCKOUT RESPONSE:", s.data)

    if (s?.data?.success) {
      stockoutRows.value = parseCategory(s.data.metrics, "risk_mae")
      stockoutGlobal.value = objToRows(s.data.global_metrics)
    }

    // 🔹 MOVEMENT
    const m = await getMovement()
    console.log("MOVEMENT RESPONSE:", m.data)

    if (m?.data?.success) {
      movementRows.value = parseCategory(m.data.metrics, "movement_mae")
      movementGlobal.value = objToRows(m.data.global_metrics)
    }

  } catch (err) {
    console.error("❌ METRICS ERROR:", err)

    toast.add({
      severity: "error",
      summary: "Metrics Failed",
      detail: err.response?.data?.error || "API request failed",
      life: 3000,
    })
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  console.log("📌 onMounted triggered")
  loadMetrics()
})
</script>

<template>
  <div class="metrics-page">

    <Toast position="top-center" />

    <h2 class="title">AI Model Metrics Dashboard</h2>

    <div v-if="loading" class="loading">
      <ProgressSpinner />
    </div>

    <div v-else class="grid">

      <!-- FORECAST -->
      <Card>
        <template #title>Demand Forecast Metrics</template>

        <DataTable :value="forecastRows" stripedRows>
          <Column field="category" header="Category" />
          <Column field="range" header="Range" />
          <Column field="mae" header="MAE" />
          <Column field="rmse" header="RMSE" />
          <Column field="mape" header="MAPE" />
        </DataTable>
      </Card>

      <!-- STOCKOUT GLOBAL -->
      <Card>
        <template #title>Stockout Risk - Global</template>

        <DataTable :value="stockoutGlobal">
          <Column field="name" header="Metric" />
          <Column field="value" header="Value" />
        </DataTable>
      </Card>

      <!-- MOVEMENT GLOBAL -->
      <Card>
        <template #title>Item Movement - Global</template>

        <DataTable :value="movementGlobal">
          <Column field="name" header="Metric" />
          <Column field="value" header="Value" />
        </DataTable>
      </Card>

      <!-- STOCKOUT CATEGORY -->
      <Card>
        <template #title>Stockout Risk By Category</template>

        <DataTable :value="stockoutRows">
          <Column field="category" header="Category" />
          <Column field="accuracy" header="Accuracy" />
          <Column field="macro_f1" header="Macro F1" />
          <Column field="risk_mae" header="Risk MAE" />
          <Column field="total_items" header="Items" />
        </DataTable>
      </Card>

      <!-- MOVEMENT CATEGORY -->
      <Card>
        <template #title>Item Movement By Category</template>

        <DataTable :value="movementRows">
          <Column field="category" header="Category" />
          <Column field="accuracy" header="Accuracy" />
          <Column field="macro_f1" header="Macro F1" />
          <Column field="movement_mae" header="MAE" />
          <Column field="total_items" header="Items" />
        </DataTable>
      </Card>

    </div>
  </div>
</template>

<style scoped>
.metrics-page { padding: 20px; }
.title { color: #fff; margin-bottom: 20px; }

.grid {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
}

.loading {
  display: flex;
  justify-content: center;
  margin-top: 40px;
}
</style>
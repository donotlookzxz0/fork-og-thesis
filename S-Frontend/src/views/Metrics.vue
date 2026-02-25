<template>
  <div class="metrics-page">

    <h2 class="title">AI Model Metrics Dashboard</h2>

    <div v-if="loading" class="loading">
      <ProgressSpinner />
    </div>

    <div v-else class="grid">

      <!-- ================= FORECAST ================= -->
      <Card>
        <template #title>Demand Forecast Metrics</template>

        <DataTable
          :value="forecastRows"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="category" header="Category" />
          <Column field="range" header="Range" />
          <Column field="mae" header="MAE" />
          <Column field="rmse" header="RMSE" />
          <Column field="mape" header="MAPE" />
        </DataTable>
      </Card>

      <!-- ================= STOCKOUT GLOBAL ================= -->
      <Card>
        <template #title>Stockout Risk - Global Metrics</template>

        <DataTable :value="stockoutGlobalRows">
          <Column field="name" header="Metric" />
          <Column field="value" header="Value" />
        </DataTable>
      </Card>

      <!-- ================= MOVEMENT GLOBAL ================= -->
      <Card>
        <template #title>Item Movement - Global Metrics</template>

        <DataTable :value="movementGlobalRows">
          <Column field="name" header="Metric" />
          <Column field="value" header="Value" />
        </DataTable>
      </Card>

      <!-- ================= STOCKOUT CATEGORY ================= -->
      <Card>
        <template #title>Stockout Risk By Category</template>

        <DataTable :value="stockoutRows" responsiveLayout="scroll">
          <Column field="category" header="Category" />
          <Column field="accuracy" header="Accuracy" />
          <Column field="macro_f1" header="Macro F1" />
          <Column field="risk_mae" header="Risk MAE" />
          <Column field="total_items" header="Items" />
        </DataTable>
      </Card>

      <!-- ================= MOVEMENT CATEGORY ================= -->
      <Card>
        <template #title>Item Movement By Category</template>

        <DataTable :value="movementRows" responsiveLayout="scroll">
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

<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

import Card from "primevue/card"
import DataTable from "primevue/datatable"
import Column from "primevue/column"
import ProgressSpinner from "primevue/progressspinner"

const loading = ref(true)

/* ================= STATE ================= */

const forecastRows = ref([])

const stockoutRows = ref([])
const stockoutGlobalRows = ref([])

const movementRows = ref([])
const movementGlobalRows = ref([])

/* ================= HELPERS ================= */

// 🔥 FLATTEN FORECAST STRUCTURE
const parseForecast = (metrics) => {
  const rows = []

  Object.keys(metrics).forEach(category => {
    const ranges = metrics[category]

    Object.keys(ranges).forEach(range => {
      const m = ranges[range]

      rows.push({
        category,
        range,
        mae: Number(m.mae).toFixed(3),
        rmse: Number(m.rmse).toFixed(3),
        mape: Number(m.mape).toFixed(2) + "%"
      })
    })
  })

  return rows
}

const objToRows = (obj) => {
  return Object.keys(obj).map(k => ({
    name: k,
    value: typeof obj[k] === "object"
      ? JSON.stringify(obj[k])
      : obj[k]
  }))
}

const parseCategoryMetrics = (metrics, maeKey) => {
  return Object.keys(metrics).map(category => ({
    category,
    accuracy: metrics[category].accuracy,
    macro_f1: metrics[category].macro_f1,
    total_items: metrics[category].total_items,
    [maeKey]: metrics[category][maeKey]
  }))
}

/* ================= API ================= */

const loadMetrics = async () => {
  try {

    // 🔹 FORECAST
    const forecast = await api.get("/metrics/forecast")

    if (forecast.data.success) {
      forecastRows.value = parseForecast(forecast.data.metrics)
    }

    // 🔹 STOCKOUT
    const stockout = await api.get("/metrics/stockout-risk")

    if (stockout.data.success) {
      stockoutRows.value = parseCategoryMetrics(
        stockout.data.metrics,
        "risk_mae"
      )

      stockoutGlobalRows.value = objToRows(
        stockout.data.global_metrics
      )
    }

    // 🔹 ITEM MOVEMENT
    const movement = await api.get("/metrics/item-movement")

    if (movement.data.success) {
      movementRows.value = parseCategoryMetrics(
        movement.data.metrics,
        "movement_mae"
      )

      movementGlobalRows.value = objToRows(
        movement.data.global_metrics
      )
    }

  } catch (err) {
    console.error("Metrics load error:", err)
  } finally {
    loading.value = false
  }
}

onMounted(loadMetrics)
</script>

<style scoped>
.metrics-page {
  padding: 20px;
}

.title {
  margin-bottom: 20px;
  font-weight: 600;
}

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
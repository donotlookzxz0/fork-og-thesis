<template>
  <div class="metrics-page">

    <h2 class="page-title">AI Model Metrics</h2>

    <!-- LOADING -->
    <div v-if="loading" class="loading-wrap">
      <ProgressSpinner />
    </div>

    <div v-else class="grid">

      <!-- FORECAST METRICS -->
      <Card class="metric-card">
        <template #title>Demand Forecast Metrics</template>

        <DataTable
          :value="formatMetrics(forecastMetrics)"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="name" header="Metric"></Column>
          <Column field="value" header="Value"></Column>
        </DataTable>
      </Card>

      <!-- STOCKOUT METRICS -->
      <Card class="metric-card">
        <template #title>Stockout Risk Metrics</template>

        <DataTable
          :value="formatMetrics(stockoutMetrics)"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="name" header="Metric"></Column>
          <Column field="value" header="Value"></Column>
        </DataTable>

        <Divider />
        <h4>Global Metrics</h4>

        <DataTable
          :value="formatMetrics(stockoutGlobal)"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="name" header="Metric"></Column>
          <Column field="value" header="Value"></Column>
        </DataTable>
      </Card>

      <!-- ITEM MOVEMENT METRICS -->
      <Card class="metric-card">
        <template #title>Item Movement Metrics</template>

        <DataTable
          :value="formatMetrics(movementMetrics)"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="name" header="Metric"></Column>
          <Column field="value" header="Value"></Column>
        </DataTable>

        <Divider />
        <h4>Global Metrics</h4>

        <DataTable
          :value="formatMetrics(movementGlobal)"
          responsiveLayout="scroll"
          stripedRows
        >
          <Column field="name" header="Metric"></Column>
          <Column field="value" header="Value"></Column>
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
import Divider from "primevue/divider"
import ProgressSpinner from "primevue/progressspinner"

const loading = ref(true)

const forecastMetrics = ref({})
const stockoutMetrics = ref({})
const stockoutGlobal = ref({})
const movementMetrics = ref({})
const movementGlobal = ref({})

/* ---------- FORMAT OBJECT → TABLE ---------- */
const formatMetrics = (obj) => {
  if (!obj) return []
  return Object.keys(obj).map(key => ({
    name: key,
    value: obj[key]
  }))
}

/* ---------- API CALLS ---------- */
const loadMetrics = async () => {
  try {
    loading.value = true

    // 🔹 Forecast
    const forecastRes = await api.get("/metrics/forecast")
    if (forecastRes.data.success) {
      forecastMetrics.value = forecastRes.data.metrics || {}
    }

    // 🔹 Stockout Risk
    const stockoutRes = await api.get("/metrics/stockout-risk")
    if (stockoutRes.data.success) {
      stockoutMetrics.value = stockoutRes.data.metrics || {}
      stockoutGlobal.value = stockoutRes.data.global_metrics || {}
    }

    // 🔹 Item Movement
    const movementRes = await api.get("/metrics/item-movement")
    if (movementRes.data.success) {
      movementMetrics.value = movementRes.data.metrics || {}
      movementGlobal.value = movementRes.data.global_metrics || {}
    }

  } catch (err) {
    console.error("Metrics load failed:", err)
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

.page-title {
  margin-bottom: 20px;
  font-weight: 600;
}

.grid {
  display: grid;
  gap: 20px;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
}

.metric-card {
  border-radius: 12px;
}

.loading-wrap {
  display: flex;
  justify-content: center;
  margin-top: 40px;
}
</style>
<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

// 🔔 PrimeVue Toast
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()

const risks = ref([])
const loading = ref(false)
const lastRun = ref(null)

// 🔄 LOAD DATA (REFRESH)
const load = async () => {
  try {
    const res = await api.get("/ml/stockout-risk")
    risks.value = res.data

    toast.add({
      severity: "success",
      summary: "Refreshed",
      detail: "Stockout risk data updated",
      life: 2500
    })

  } catch (err) {
    console.error("Load failed:", err)
    toast.add({
      severity: "error",
      summary: "Error",
      detail: "Failed to load stockout risk data",
      life: 3000
    })
  }
}

// 🤖 RUN MODEL
const run = async () => {
  try {
    loading.value = true

    toast.add({
      severity: "info",
      summary: "Running Model",
      detail: "Stockout risk analysis in progress...",
      life: 2000
    })

    await api.post("/ml/stockout-risk")
    await load()

    lastRun.value = new Date().toLocaleString()

    toast.add({
      severity: "success",
      summary: "Completed",
      detail: "Stockout risk forecast updated",
      life: 3000
    })

  } catch (err) {
    console.error("Run failed:", err)
    toast.add({
      severity: "error",
      summary: "AI Error",
      detail: "Failed to run stockout risk model",
      life: 3000
    })
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="page">

    <!-- 🔔 TOAST -->
    <Toast position="top-center" />

    <h1 class="title"><i class="pi pi-exclamation-triangle"></i> Stockout Risk</h1>

    <!-- CONTROLS -->
    <div class="controls">
      <!-- 🔄 REFRESH -->
      <button class="btn secondary" @click="load" :disabled="loading">
        <i class="pi pi-refresh"></i> Refresh
      </button>

      <!-- 🤖 RUN MODEL -->
      <button class="btn primary" @click="run" :disabled="loading">
        <i class="pi pi-cog"></i>
        {{ loading ? "Running Model..." : "Run Model" }}
      </button>

      <span v-if="lastRun" class="timestamp">
        Last run: {{ lastRun }}
      </span>
    </div>

    <!-- TABLE CARD -->
    <div class="card">
      <table class="risk-table">
        <thead>
          <tr>
            <th>Item</th>
            <th>Category</th>
            <th>Current Stock</th>
            <th>Risk Level</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in risks" :key="r.item_id">
            <td>{{ r.item_name }}</td>
            <td>{{ r.category }}</td>
            <td>{{ r.current_stock }}</td>
            <td>
              <span class="badge" :class="r.risk_level.toLowerCase()">
                {{ r.risk_level }}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- ⏳ LOADER OVERLAY -->
    <div v-if="loading" class="loader-overlay">
      <div class="loader-box">
        <i class="pi pi-spin pi-spinner"></i>
        <span>Running Stockout Risk Analysis...</span>
      </div>
    </div>

  </div>
</template>

<style scoped>
.page { padding: 20px; }

/* Title */
.title {
  color: #ffffff;
  font-size: 1.8rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

/* Controls */
.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}

.timestamp { font-size: 0.85rem; color: #aaa; }

/* Buttons */
.btn {
  height: 48px;
  padding: 0 18px;
  font-size: 1rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
}

.primary {
  background: #3ddc97;
  border: none;
}

.secondary {
  background: #444;
  color: #fff;
  border: none;
}

/* Card */
.card {
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 16px;
}

/* Table */
.risk-table {
  width: 100%;
  border-collapse: collapse;
}

.risk-table th,
.risk-table td {
  border: 1px solid #333;
  padding: 10px;
  text-align: center;
}

.risk-table th {
  font-size: 0.85rem;
  color: #aaa;
}

.risk-table tr:hover {
  background: rgba(255,255,255,0.03);
}

/* Risk Badges */
.badge {
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: bold;
}

.badge.low { background: #27ae60; }
.badge.medium { background: #f39c12; }
.badge.high { background: #c0392b; }

/* ⏳ Loader */
.loader-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.6);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 999;
}

.loader-box {
  background: #1f1f1f;
  border: 1px solid #333;
  padding: 24px 36px;
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
  color: #fff;
  box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.loader-box i {
  font-size: 2rem;
  color: #3ddc97;
}
</style>

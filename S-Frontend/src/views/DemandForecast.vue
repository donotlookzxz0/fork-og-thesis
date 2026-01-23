<script setup>
import { ref, onMounted, watch, computed, nextTick } from "vue"
import api from "../services/api"
import { Chart } from "chart.js/auto"

/* =========================
   STATE
========================= */
const categories = ref([])
const selectedCategory = ref("")
const forecast = ref(null)

const demand = ref({
  tomorrow: 0,
  next7: 0,
  next30: 0
})

const lastRun = ref(null)
const loading = ref(false)

/* =========================
   CHART STATE
========================= */
const horizon = ref("30") // 1 | 7 | 30
const canvas = ref(null)
let chart = null

/* =========================
   LOAD FORECAST
========================= */
const loadForecast = async () => {
  try {
    const res = await api.get("/ml/forecast")
    forecast.value = res.data
    categories.value = forecast.value.tomorrow.map(f => f.category)
    await nextTick()
    renderChart()
  } catch (err) {
    console.error("Load forecast failed:", err)
  }
}

/* =========================
   RUN AI MODEL (SAFE)
========================= */
const runAllModels = async () => {
  try {
    loading.value = true

    // ✅ ONLY ONE API CALL (Render-safe)
    await api.post("/ml/forecast")

    lastRun.value = new Date().toLocaleString()
    await loadForecast()
    computeDemand()
  } catch (err) {
    console.error("AI run failed:", err)
  } finally {
    loading.value = false
  }
}

/* =========================
   COMPUTE DEMAND
========================= */
const computeDemand = () => {
  if (!forecast.value || !selectedCategory.value) return

  const t = forecast.value.tomorrow.find(f => f.category === selectedCategory.value)
  const d7 = forecast.value.next_7_days.find(f => f.category === selectedCategory.value)
  const d30 = forecast.value.next_30_days.find(f => f.category === selectedCategory.value)

  demand.value = {
    tomorrow: t?.predicted_quantity ?? 0,
    next7: d7?.predicted_quantity ?? 0,
    next30: d30?.predicted_quantity ?? 0
  }
}

/* =========================
   DERIVED DATA
========================= */
const demandLevel = computed(() => {
  if (demand.value.next30 > 15) return "High"
  if (demand.value.next30 >= 5) return "Medium"
  return "Low"
})

const suggestions = computed(() => {
  if (!selectedCategory.value) return []
  if (demandLevel.value === "High") {
    return [
      "Prepare restock within 7 days",
      "Confirm supplier availability",
      "Monitor inventory daily"
    ]
  }
  if (demandLevel.value === "Medium") {
    return [
      "Monitor weekly sales trend",
      "Plan replenishment if needed"
    ]
  }
  return [
    "No immediate restocking required",
    "Consider promotional strategies"
  ]
})

/* =========================
   TOP DEMAND
========================= */
const topDemand = computed(() => {
  if (!forecast.value) return []

  return forecast.value.next_30_days
    .map(c => {
      const t = forecast.value.tomorrow.find(x => x.category === c.category)
      const d7 = forecast.value.next_7_days.find(x => x.category === c.category)

      return {
        category: c.category,
        tomorrow: t?.predicted_quantity ?? 0,
        next7: d7?.predicted_quantity ?? 0,
        next30: c.predicted_quantity
      }
    })
    .sort((a, b) => b.next30 - a.next30)
    .slice(0, 5)
})

/* =========================
   CHART DATA
========================= */
const chartData = computed(() => {
  if (!forecast.value) return []
  if (horizon.value === "1") return forecast.value.tomorrow
  if (horizon.value === "7") return forecast.value.next_7_days
  return forecast.value.next_30_days
})

const renderChart = () => {
  if (!canvas.value || !chartData.value.length) return
  if (chart) chart.destroy()

  const labels = chartData.value.map(c => c.category)
  const values = chartData.value.map(c => c.predicted_quantity)
  const total = values.reduce((a, b) => a + b, 0)

  chart = new Chart(canvas.value, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [
          "#3498db","#9b59b6","#1abc9c","#f1c40f",
          "#e67e22","#e74c3c","#2ecc71","#95a5a6",
          "#16a085","#8e44ad","#2980b9"
        ]
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#ccc", boxWidth: 12 }
        },
        tooltip: {
          callbacks: {
            label: ctx => {
              const value = ctx.raw
              const pct = total ? ((value / total) * 100).toFixed(1) : 0
              return `${ctx.label}: ${value} (${pct}%)`
            }
          }
        }
      }
    }
  })
}

watch(selectedCategory, computeDemand)
watch(horizon, renderChart)
onMounted(loadForecast)
</script>

<template>
  <div class="page">
    <h1>Demand Forecast (Category-Based)</h1>

    <!-- CONTROLS -->
    <div class="controls">
      <select v-model="selectedCategory">
        <option value="">-- Select Category --</option>
        <option v-for="c in categories" :key="c" :value="c">
          {{ c }}
        </option>
      </select>

      <button @click="runAllModels" :disabled="loading">
        {{ loading ? "Running AI..." : "Run AI Forecast" }}
      </button>

      <span v-if="lastRun" class="timestamp">
        Last run: {{ lastRun }}
      </span>
    </div>

    <!-- LAYOUT -->
    <div class="layout">
      <!-- LEFT -->
      <div class="left">
        <div v-if="selectedCategory" class="card">
          <h2>
            {{ selectedCategory }}
            <span class="badge" :class="demandLevel.toLowerCase()">
              {{ demandLevel }} Demand
            </span>
          </h2>

          <div class="stats">
            <div class="stat"><span>Tomorrow</span><strong>{{ demand.tomorrow }}</strong></div>
            <div class="stat"><span>Next 7 Days</span><strong>{{ demand.next7 }}</strong></div>
            <div class="stat"><span>Next 30 Days</span><strong>{{ demand.next30 }}</strong></div>
          </div>
        </div>

        <div v-if="selectedCategory" class="card">
          <h3>Suggested Actions</h3>
          <ul>
            <li v-for="s in suggestions" :key="s">{{ s }}</li>
          </ul>
        </div>

        <div class="card">
          <h3>Top Demand Categories</h3>
          <table class="top-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Tomorrow</th>
                <th>Next 7 Days</th>
                <th>Next 30 Days</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="c in topDemand" :key="c.category">
                <td>{{ c.category }}</td>
                <td>{{ c.tomorrow }}</td>
                <td>{{ c.next7 }}</td>
                <td class="emphasis">{{ c.next30 }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- RIGHT -->
      <div class="right">
        <div class="card chart-card">
          <div class="chart-header">
            <h3>Category Demand Distribution</h3>
            <select v-model="horizon">
              <option value="1">Tomorrow</option>
              <option value="7">Next 7 Days</option>
              <option value="30">Next 30 Days</option>
            </select>
          </div>

          <div class="chart-wrapper">
            <canvas ref="canvas"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 20px; }

.controls {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 20px;
}

.timestamp { font-size: 0.85rem; color: #aaa; }

.layout {
  display: grid;
  grid-template-columns: 1.3fr 1fr;
  gap: 24px;
}

.card {
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 20px;
}

.chart-card {
  height: 360px;
}

.chart-wrapper {
  height: 260px;
}

.badge {
  margin-left: 10px;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 0.75rem;
}
.badge.high { background: #c0392b; }
.badge.medium { background: #f39c12; }
.badge.low { background: #27ae60; }

.stats {
  display: flex;
  justify-content: space-between;
  margin-top: 16px;
}
.stat span { font-size: 0.85rem; color: #aaa; }
.stat strong { font-size: 1.8rem; }

.top-table {
  width: 100%;
  border-collapse: collapse;
}
.top-table th, .top-table td {
  padding: 8px;
  border-bottom: 1px solid #333;
  text-align: center;
}
.top-table th { font-size: 0.85rem; color: #aaa; }
.emphasis { font-weight: bold; }

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
</style>

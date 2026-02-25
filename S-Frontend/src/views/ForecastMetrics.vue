<script setup>
import { ref, onMounted } from "vue"
import { useRouter } from "vue-router"
import api from "../services/api"

const router = useRouter()

const loading = ref(true)
const rows = ref([])
const global = ref({
  avg_mae: 0,
  avg_rmse: 0,
  avg_mape: 0
})

/* ---------------- GO BACK ---------------- */
const goBack = () => {
  router.back()
}

/* ---------------- PARSE METRICS ---------------- */
const parseMetrics = (metrics) => {
  const list = []

  let maeSum = 0
  let rmseSum = 0
  let mapeSum = 0
  let count = 0

  Object.keys(metrics).forEach(category => {
    const ranges = metrics[category]

    Object.keys(ranges).forEach(range => {
      const m = ranges[range]

      list.push({
        category,
        range,
        mae: Number(m.mae),
        rmse: Number(m.rmse),
        mape: Number(m.mape)
      })

      maeSum += Number(m.mae)
      rmseSum += Number(m.rmse)
      mapeSum += Number(m.mape)
      count++
    })
  })

  // 🔥 Compute GLOBAL KPI values
  global.value.avg_mae = (maeSum / count).toFixed(2)
  global.value.avg_rmse = (rmseSum / count).toFixed(2)
  global.value.avg_mape = (mapeSum / count).toFixed(2)

  return list
}

/* ---------------- LOAD DATA ---------------- */
const load = async () => {
  try {
    const res = await api.get("/metrics/forecast")

    if (res.data.success) {
      rows.value = parseMetrics(res.data.metrics)
    }
  } catch (err) {
    console.error("Forecast metrics error:", err)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="metrics-page">

    <!-- 🔙 GO BACK BUTTON -->
    <button class="back-btn" @click="goBack">
      ← Go Back
    </button>

    <h2 class="title">Demand Forecast AI Metrics</h2>

    <div v-if="loading" class="loading">
      Loading model metrics...
    </div>

    <div v-else>

      <!-- 🔥 KPI CARDS -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <span>Average MAE</span>
          <h3>{{ global.avg_mae }}</h3>
        </div>

        <div class="kpi-card">
          <span>Average RMSE</span>
          <h3>{{ global.avg_rmse }}</h3>
        </div>

        <div class="kpi-card">
          <span>Average MAPE</span>
          <h3>{{ global.avg_mape }}%</h3>
        </div>
      </div>

      <!-- 📊 CATEGORY TABLE -->
      <div class="table-wrapper">
        <h3>Forecast Accuracy by Category</h3>

        <table>
          <thead>
            <tr>
              <th>Category</th>
              <th>Range</th>
              <th>MAE</th>
              <th>RMSE</th>
              <th>MAPE</th>
            </tr>
          </thead>

          <tbody>
            <tr v-for="r in rows" :key="r.category + r.range">
              <td>{{ r.category }}</td>
              <td>{{ r.range }}</td>

              <td>{{ r.mae.toFixed(2) }}</td>
              <td>{{ r.rmse.toFixed(2) }}</td>

              <td :class="r.mape < 40 ? 'good' : r.mape < 70 ? 'warn' : 'bad'">
                {{ r.mape.toFixed(2) }}%
              </td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  </div>
</template>

<style scoped>
.metrics-page {
  padding: 20px;
  color: #fff;
}

.title {
  margin-bottom: 20px;
  font-size: 1.8rem;
}

/* 🔙 BACK BUTTON */
.back-btn {
  margin-bottom: 12px;
  padding: 8px 14px;
  border-radius: 6px;
  border: none;
  background: #1f2937;
  color: #fff;
  cursor: pointer;
}

.back-btn:hover {
  background: #374151;
}

/* 🔥 KPI GRID */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.kpi-card {
  background: #111827;
  border-radius: 10px;
  padding: 18px;
  box-shadow: 0 0 10px rgba(0,0,0,0.3);
}

.kpi-card span {
  font-size: 0.9rem;
  color: #9ca3af;
}

.kpi-card h3 {
  margin-top: 6px;
  font-size: 1.6rem;
}

/* 📊 TABLE */
.table-wrapper {
  background: #0f172a;
  border-radius: 10px;
  padding: 16px;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th {
  text-align: left;
  padding: 10px;
  background: #020617;
}

td {
  padding: 10px;
  border-bottom: 1px solid #1f2937;
}

.good {
  color: #22c55e;
  font-weight: bold;
}

.warn {
  color: #f59e0b;
  font-weight: bold;
}

.bad {
  color: #ef4444;
  font-weight: bold;
}

.loading {
  margin-top: 40px;
}
</style>
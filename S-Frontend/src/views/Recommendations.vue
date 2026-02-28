<script setup>
import { ref, onMounted, computed } from "vue"
import api from "../services/api"

import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()

const recommendations = ref([])
const totalUsers = ref(0)
const loading = ref(false)
const training = ref(false)
const trainingLogs = ref([])
const metrics = ref({
  rmse: null,
  mse: null
})

const usersWithRecs = computed(() => {
  return recommendations.value.filter(r => r.recommendations && r.recommendations.length > 0).length
})

const usersWithoutRecs = computed(() => {
  return totalUsers.value - usersWithRecs.value
})

const loadRecommendations = async () => {
  try {
    loading.value = true
    const [recsRes, usersRes] = await Promise.all([
      api.get("/recommendations"),
      api.get("/users")
    ])

    recommendations.value = recsRes.data
    totalUsers.value = usersRes.data.length

  } catch (err) {
    console.error("Load recommendations failed:", err)
    toast.add({
      severity: "error",
      summary: "Load Failed",
      detail: err.response?.data?.error || "Failed to load recommendations",
      life: 3000,
    })
  } finally {
    loading.value = false
  }
}

const runRecommender = async () => {
  if (training.value) {
    toast.add({
      severity: "warn",
      summary: "Already Running",
      detail: "Recommender is currently training",
      life: 2000,
    })
    return
  }

  try {
    training.value = true
    trainingLogs.value = ["Training..."]

    const res = await api.post("/recommendations/train")

    if (res.data.success) {
      trainingLogs.value = res.data.logs || []

      metrics.value = {
        rmse: res.data.rmse,
        mse: res.data.mse
      }

      toast.add({
        severity: "success",
        summary: "Training Completed",
        detail: "AI model updated successfully",
        life: 3000,
      })

      await loadRecommendations()
    }

  } catch (err) {
    console.error("Recommender run failed:", err)
    toast.add({
      severity: "error",
      summary: "Training Failed",
      detail: err.response?.data?.error || "Failed to run recommender",
      life: 3000,
    })
  } finally {
    training.value = false
  }
}

onMounted(async () => {
  await loadRecommendations()
})
</script>

<template>
  <div class="page">
    <Toast position="top-center" />

    <h2 class="title">AI Recommendations</h2>

    <div class="controls">
      <button class="btn primary" @click="runRecommender" :disabled="training">
        <i class="pi pi-refresh"></i>
        {{ training ? "Training..." : "Update Recommendations" }}
      </button>
    </div>

    <div v-if="loading" class="loading">
      Loading recommendations...
    </div>

    <div v-else>
      <div class="kpi-grid">
        <div class="kpi-card">
          <span>Total Users</span>
          <h3>{{ totalUsers }}</h3>
        </div>

        <div class="kpi-card">
          <span>Users With Recommendations</span>
          <h3>{{ usersWithRecs }}</h3>
        </div>

        <div class="kpi-card">
          <span>Users Without Recommendations</span>
          <h3>{{ usersWithoutRecs }}</h3>
        </div>

        <div class="kpi-card" v-if="metrics.rmse !== null">
          <span>RMSE</span>
          <h3>{{ metrics.rmse }}</h3>
        </div>

        <div class="kpi-card" v-if="metrics.mse !== null">
          <span>MSE</span>
          <h3>{{ metrics.mse }}</h3>
        </div>
      </div>

      <div v-if="trainingLogs.length" class="logs-wrapper">
        <h3>Training Logs</h3>
        <div class="logs">
          <div v-for="(log, idx) in trainingLogs" :key="idx" class="log-entry">
            {{ log }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  padding: 20px;
  color: #fff;
}

.title {
  margin-bottom: 20px;
  font-size: 1.8rem;
}

.controls {
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 20px;
}

.btn {
  height: 48px;
  padding: 0 18px;
  font-size: 1rem;
  cursor: pointer;
  border: none;
  display: flex;
  align-items: center;
  gap: 8px;
  border-radius: 6px;
}

.primary {
  background: #3ddc97;
  color: #000;
}

.primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

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
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
}

.kpi-card span {
  font-size: 0.9rem;
  color: #9ca3af;
}

.kpi-card h3 {
  margin-top: 6px;
  font-size: 1.6rem;
}

.logs-wrapper {
  background: #0f172a;
  border-radius: 10px;
  padding: 16px;
  margin-top: 24px;
}

.logs-wrapper h3 {
  margin-bottom: 12px;
  font-size: 1.2rem;
}

.logs {
  background: #020617;
  border-radius: 6px;
  padding: 12px;
  font-family: monospace;
  font-size: 0.9rem;
  max-height: 300px;
  overflow-y: auto;
}

.log-entry {
  padding: 4px 0;
  color: #9ca3af;
}

.loading {
  margin-top: 40px;
  color: #9ca3af;
}
</style>
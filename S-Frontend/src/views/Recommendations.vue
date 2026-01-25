<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

// 🔔 PrimeVue Toast
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()

const recommendations = ref([])
const loading = ref(false)
const lastRun = ref(null)

/* =========================
   LOAD RECOMMENDATIONS
========================= */
const loadRecommendations = async () => {
  try {
    const res = await api.get("/recommendations")
    recommendations.value = res.data

    // ✅ LOADED SUCCESS
    toast.add({
      severity: "info",
      summary: "Recommendations Loaded",
      detail: "Latest recommendations retrieved",
      life: 2000,
    })

  } catch (err) {
    console.error("Load recommendations failed:", err)

    toast.add({
      severity: "error",
      summary: "Load Failed",
      detail: err.response?.data?.error || "Failed to load recommendations",
      life: 3000,
    })
  }
}

/* =========================
   RUN RECOMMENDER (ADMIN)
========================= */
const runRecommender = async () => {
  if (loading.value) {
    toast.add({
      severity: "warn",
      summary: "Already Running",
      detail: "Recommender is currently training",
      life: 2000,
    })
    return
  }

  try {
    loading.value = true

    await api.post("/recommendations/train")
    lastRun.value = new Date().toLocaleString()

    // ✅ TRAINING SUCCESS
    toast.add({
      severity: "success",
      summary: "Training Completed",
      detail: "AI model updated successfully",
      life: 3000,
    })

    await loadRecommendations()

  } catch (err) {
    console.error("Recommender run failed:", err)

    toast.add({
      severity: "error",
      summary: "Training Failed",
      detail: err.response?.data?.error || "Failed to run recommender",
      life: 3000,
    })

  } finally {
    loading.value = false
  }
}

onMounted(loadRecommendations)
</script>

<template>
  <div class="page">

    <!-- 🔔 TOAST POPUPS -->
    <Toast position="top-center" />

    <!-- 🔥 TITLE MATCHING YOUR ADMIN STYLE -->
    <h1 class="title">
      <i class="pi pi-brain"></i>
      AI Recommendations
    </h1>

    <div class="controls">
      <button
        class="btn primary"
        @click="runRecommender"
        :disabled="loading"
      >
        <i class="pi pi-refresh"></i>
        {{ loading ? "Training Recommender..." : "Update Recommendations" }}
      </button>

      <span v-if="lastRun" class="timestamp">
        Last run: {{ lastRun }}
      </span>
    </div>

    <div v-if="!recommendations.length" class="empty">
      No recommendations yet. Run the AI model.
    </div>

    <div class="grid">
      <div v-for="r in recommendations" :key="r.user_id" class="card">
        <h3>User #{{ r.user_id }}</h3>

        <ul>
          <li v-for="item in r.items" :key="item.id">
            {{ item.name }}
            <span class="score">({{ item.score.toFixed(2) }})</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  padding: 20px;
}

/* 🔥 MATCH TITLE STYLE WITH OTHER PAGES */
.title {
  color: #ffffff;
  font-size: 1.8rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

/* CONTROLS */
.controls {
  display: flex;
  gap: 14px;
  align-items: center;
  margin-bottom: 20px;
}

.timestamp {
  font-size: 0.85rem;
  color: #aaa;
}

/* BUTTON STYLE (MATCH INVENTORY / WALLET) */
.btn {
  height: 48px;
  padding: 0 18px;
  font-size: 1rem;
  cursor: pointer;
  border: none;
  display: flex;
  align-items: center;
  gap: 8px;
}

.primary {
  background: #3ddc97;
  color: #000;
}

/* GRID */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

/* CARD */
.card {
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 14px;
}

/* SCORE */
.score {
  color: #aaa;
  font-size: 0.8rem;
}

/* EMPTY STATE */
.empty {
  color: #9ca3af;
  padding: 40px 0;
  text-align: center;
}
</style>

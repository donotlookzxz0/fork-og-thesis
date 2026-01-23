<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

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
  } catch (err) {
    console.error("Load recommendations failed:", err)
  }
}

/* =========================
   RUN RECOMMENDER (ADMIN)
========================= */
const runRecommender = async () => {
  try {
    loading.value = true
    await api.post("/recommendations/train")
    lastRun.value = new Date().toLocaleString()
    await loadRecommendations()
  } catch (err) {
    console.error("Recommender run failed:", err)
  } finally {
    loading.value = false
  }
}

onMounted(loadRecommendations)
</script>

<template>
  <div class="page">
    <h1>AI Recommendations</h1>

    <div class="controls">
      <button @click="runRecommender" :disabled="loading">
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
.page { padding: 20px; }
.controls { display: flex; gap: 12px; margin-bottom: 20px; }
.timestamp { font-size: 0.85rem; color: #aaa; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 16px;
}

.card {
  background: #1f1f1f;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 14px;
}

.score {
  color: #aaa;
  font-size: 0.8rem;
}
</style>
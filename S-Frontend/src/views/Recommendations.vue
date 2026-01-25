<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

// 🔔 PrimeVue Toast
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()

const recommendations = ref([])
const usersMap = ref({})   // 🔑 id -> username lookup
const loading = ref(false)
const lastRun = ref(null)

/* =========================
   LOAD USERS (FOR NAME MAP)
========================= */
const loadUsers = async () => {
  try {
    const res = await api.get("/users")

    // Build { id: username } map
    usersMap.value = res.data.reduce((map, u) => {
      map[u.id] = u.username
      return map
    }, {})

  } catch (err) {
    console.error("Load users failed:", err)

    toast.add({
      severity: "error",
      summary: "User Load Failed",
      detail: "Failed to load user names",
      life: 3000,
    })
  }
}

/* =========================
   LOAD RECOMMENDATIONS
========================= */
const loadRecommendations = async () => {
  try {
    const res = await api.get("/recommendations")
    recommendations.value = res.data

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

/* =========================
   INIT
========================= */
onMounted(async () => {
  await loadUsers()           // 🔑 load usernames first
  await loadRecommendations()
})
</script>

<template>
  <div class="page">

    <!-- 🔔 TOAST POPUPS -->
    <Toast position="top-center" />

    <!-- 🔥 TITLE -->
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

        <!-- 🔥 USER NAME INSTEAD OF ID -->
        <h3>
          {{ usersMap[r.user_id] || `User #${r.user_id}` }}
        </h3>

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

/* TITLE STYLE */
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

/* BUTTON */
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

/* EMPTY */
.empty {
  color: #9ca3af;
  padding: 40px 0;
  text-align: center;
}
</style>

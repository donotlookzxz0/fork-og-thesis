<script setup>
import { ref, onMounted, watch, computed } from "vue";
import api from "../services/api";
import { useRouter } from "vue-router";

import "../assets/DemandForecast.css";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

const toast = useToast();
const router = useRouter();

const categories = ref([]);
const selectedCategory = ref("");
const forecast = ref(null);

const demand = ref({
  tomorrow: 0,
  next7: 0,
  next30: 0,
});

const lastRun = ref(null);
const loading = ref(false);
const horizon = ref("30");

const categoryColors = {
  Fruits: "#ff6b6b",
  Vegetables: "#2ecc71",
  Meat: "#c0392b",
  Seafood: "#1abc9c",
  Dairy: "#f1c40f",
  Beverages: "#3498db",
  Snacks: "#e67e22",
  Bakery: "#d35400",
  Frozen: "#9b59b6",
  "Canned Goods": "#95a5a6",
  Condiments: "#e84393",
  "Dry Goods": "#7f8c8d",
  "Grains & Pasta": "#27ae60",
  "Spices & Seasonings": "#8e44ad",
  "Breakfast & Cereal": "#f39c12",
  "Personal Care": "#fd79a8",
  Household: "#636e72",
  "Baby Products": "#fab1a0",
  "Pet Supplies": "#00cec9",
  "Health & Wellness": "#55efc4",
  "Cleaning Supplies": "#0984e3"
};

const goToMetrics = () => {
  router.push("/analytics/forecast-metrics");
};

const loadForecast = async () => {
  try {
    const res = await api.get("/ml/forecast");
    forecast.value = res.data;

    const allCategories = [
      ...forecast.value.tomorrow,
      ...forecast.value.next_7_days,
      ...forecast.value.next_30_days,
    ].map(c => c.category);

    categories.value = [...new Set(allCategories)];

    toast.add({
      severity: "success",
      summary: "Refreshed",
      detail: "Forecast data updated",
      life: 2500,
    });
  } catch (err) {
    console.error("Load forecast failed:", err);
  }
};

const runAllModels = async () => {
  try {
    loading.value = true;
    await api.post("/ml/forecast");
    lastRun.value = new Date().toLocaleString();
    await loadForecast();
    computeDemand();
  } catch (err) {
    console.error("AI run failed:", err);
  } finally {
    loading.value = false;
  }
};

const computeDemand = () => {
  if (!forecast.value || !selectedCategory.value) return;

  const t = forecast.value.tomorrow.find(
    (f) => f.category === selectedCategory.value
  );
  const d7 = forecast.value.next_7_days.find(
    (f) => f.category === selectedCategory.value
  );
  const d30 = forecast.value.next_30_days.find(
    (f) => f.category === selectedCategory.value
  );

  demand.value = {
    tomorrow: t?.predicted_quantity ?? 0,
    next7: d7?.predicted_quantity ?? 0,
    next30: d30?.predicted_quantity ?? 0,
  };
};

const demandLevel = computed(() => {
  if (demand.value.next30 > 15) return "High";
  if (demand.value.next30 >= 5) return "Medium";
  return "Low";
});

const suggestions = computed(() => {
  if (!selectedCategory.value) return [];

  if (demandLevel.value === "High") {
    return [
      "Prepare restock within 7 days",
      "Confirm supplier availability",
      "Monitor inventory daily",
    ];
  }

  if (demandLevel.value === "Medium") {
    return ["Monitor weekly sales trend", "Plan replenishment if needed"];
  }

  return [
    "No immediate restocking required",
    "Consider promotional strategies",
  ];
});

const topDemand = computed(() => {
  if (!forecast.value) return [];

  return categories.value
    .map((cat) => {
      const t = forecast.value.tomorrow.find(x => x.category === cat);
      const d7 = forecast.value.next_7_days.find(x => x.category === cat);
      const d30 = forecast.value.next_30_days.find(x => x.category === cat);

      return {
        category: cat,
        tomorrow: t?.predicted_quantity ?? 0,
        next7: d7?.predicted_quantity ?? 0,
        next30: d30?.predicted_quantity ?? 0,
      };
    })
    .sort((a,b) => b.next30 - a.next30)
    .slice(0,5);
});

const chartData = computed(() => {
  if (!forecast.value) return [];

  const source =
    horizon.value === "1"
      ? forecast.value.tomorrow
      : horizon.value === "7"
      ? forecast.value.next_7_days
      : forecast.value.next_30_days;

  return categories.value
    .map(cat => {
      const found = source.find(x => x.category === cat);
      return {
        category: cat,
        value: found?.predicted_quantity ?? 0,
        color: categoryColors[cat] || "#888"
      };
    })
    .sort((a,b) => b.value - a.value);
});

watch(selectedCategory, computeDemand);

onMounted(loadForecast);
</script>

<template>
  <div class="page">
    <Toast position="top-center" />

    <h1 class="title"><i class="pi pi-chart-pie"></i> Demand Forecast</h1>

    <div class="controls">
      <select v-model="selectedCategory">
        <option value="">-- Select Category --</option>
        <option v-for="c in categories" :key="c" :value="c">
          {{ c }}
        </option>
      </select>

      <button class="btn secondary" @click="loadForecast" :disabled="loading">
        <i class="pi pi-refresh"></i> Refresh
      </button>

      <button class="btn primary" @click="runAllModels" :disabled="loading">
        <i class="pi pi-cog"></i>
        {{ loading ? "Running AI..." : "Run AI Forecast" }}
      </button>

      <!-- NEW METRICS BUTTON -->
      <button class="btn secondary" @click="goToMetrics">
        <i class="pi pi-chart-line"></i> Metrics
      </button>

      <span v-if="lastRun" class="timestamp"> Last run: {{ lastRun }} </span>
    </div>

    <div class="layout">
      <div class="left">
        <div v-if="selectedCategory" class="card">
          <h2>
            {{ selectedCategory }}
            <span class="badge" :class="demandLevel.toLowerCase()">
              {{ demandLevel }} Demand
            </span>
          </h2>

          <div class="stats">
            <div class="stat">
              <span>Tomorrow</span><strong>{{ demand.tomorrow }} pcs</strong>
            </div>
            <div class="stat">
              <span>Next 7 Days</span><strong>{{ demand.next7 }} pcs</strong>
            </div>
            <div class="stat">
              <span>Next 30 Days</span><strong>{{ demand.next30 }} pcs</strong>
            </div>
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
                <td>{{ c.tomorrow }} pcs</td>
                <td>{{ c.next7 }} pcs</td>
                <td class="emphasis">{{ c.next30 }} pcs</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

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
            <div v-for="row in chartData" :key="row.category" style="margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;font-size:12px;color:#ccc;margin-bottom:4px;">
                <span>{{ row.category }}</span>
                <span>{{ row.value }} pcs</span>
              </div>
              <div style="background:#2a2a2a;height:14px;border-radius:6px;overflow:hidden;">
                <div
                  :style="{
                    width: (row.value / (chartData[0]?.value || 1) * 100) + '%',
                    background: row.color,
                    height:'100%',
                    borderRadius:'6px'
                  }"
                ></div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <div v-if="loading" class="loader-overlay">
      <div class="loader-box">
        <i class="pi pi-spin pi-spinner"></i>
        <span>Running AI Forecast...</span>
      </div>
    </div>
  </div>
</template>
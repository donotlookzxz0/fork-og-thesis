<script setup>
import { ref, onMounted, watch, computed, nextTick } from "vue";
import api from "../services/api";
import { Chart } from "chart.js/auto";

import "@/assets/DemandForecast.css";
import Toast from "primevue/toast";
import { useToast } from "primevue/usetoast";

const toast = useToast();

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
const canvas = ref(null);
let chart = null;

// 🔄 LOAD FORECAST
const loadForecast = async () => {
  try {
    const res = await api.get("/ml/forecast");
    forecast.value = res.data;
// ✅ Merge all horizons so categories never get skipped
const allCategories = [
  ...forecast.value.tomorrow,
  ...forecast.value.next_7_days,
  ...forecast.value.next_30_days,
].map(c => c.category);

categories.value = [...new Set(allCategories)];
    await nextTick();
    renderChart();

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

// 🤖 RUN AI
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

// 📊 COMPUTE DEMAND
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

// 🔥 DEMAND LEVEL
const demandLevel = computed(() => {
  if (demand.value.next30 > 15) return "High";
  if (demand.value.next30 >= 5) return "Medium";
  return "Low";
});

// 💡 SUGGESTIONS
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

// ⭐ TOP DEMAND
const topDemand = computed(() => {
  if (!forecast.value) return [];

  return forecast.value.next_30_days
    .map((c) => {
      const t = forecast.value.tomorrow.find((x) => x.category === c.category);
      const d7 = forecast.value.next_7_days.find(
        (x) => x.category === c.category
      );

      return {
        category: c.category,
        tomorrow: t?.predicted_quantity ?? 0,
        next7: d7?.predicted_quantity ?? 0,
        next30: c.predicted_quantity,
      };
    })
    .sort((a, b) => b.next30 - a.next30)
    .slice(0, 5);
});

// 📈 CHART DATA
const chartData = computed(() => {
  if (!forecast.value) return [];
  if (horizon.value === "1") return forecast.value.tomorrow;
  if (horizon.value === "7") return forecast.value.next_7_days;
  return forecast.value.next_30_days;
});

// 🟢 NEW PRO HORIZONTAL BAR CHART
const renderChart = () => {
  if (!canvas.value || !chartData.value.length) return;
  if (chart) chart.destroy();

  // 🔥 Sort like football table
  const sorted = [...chartData.value].sort(
    (a, b) => b.predicted_quantity - a.predicted_quantity
  );

  const labels = sorted.map((c) => c.category);
  const values = sorted.map((c) => c.predicted_quantity);

  chart = new Chart(canvas.value, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          data: values,
          borderRadius: 6,
          barThickness: 18,
          backgroundColor: [
            "#3498db",
            "#9b59b6",
            "#1abc9c",
            "#f1c40f",
            "#e67e22",
            "#e74c3c",
            "#2ecc71",
            "#95a5a6",
            "#16a085",
            "#8e44ad",
            "#2980b9",
          ],
        },
      ],
    },
    options: {
      indexAxis: "y", // ← makes it horizontal
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.raw} pcs`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#aaa" },
        },
        y: {
          grid: { display: false },
          ticks: { color: "#ccc" },
        },
      },
    },
  });
};

watch(selectedCategory, computeDemand);
watch(horizon, renderChart);
onMounted(loadForecast);
</script>

<template>
  <div class="page">
    <!-- 🔔 TOAST -->
    <Toast position="top-center" />

    <h1 class="title"><i class="pi pi-chart-pie"></i> Demand Forecast</h1>

    <!-- CONTROLS -->
    <div class="controls">
      <select v-model="selectedCategory">
        <option value="">-- Select Category --</option>
        <option v-for="c in categories" :key="c" :value="c">
          {{ c }}
        </option>
      </select>

      <!-- 🔄 REFRESH -->
      <button class="btn secondary" @click="loadForecast" :disabled="loading">
        <i class="pi pi-refresh"></i> Refresh
      </button>

      <!-- 🤖 RUN AI -->
      <button class="btn primary" @click="runAllModels" :disabled="loading">
        <i class="pi pi-cog"></i>
        {{ loading ? "Running AI..." : "Run AI Forecast" }}
      </button>

      <span v-if="lastRun" class="timestamp"> Last run: {{ lastRun }} </span>
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

    <!-- ⏳ LOADER OVERLAY -->
    <div v-if="loading" class="loader-overlay">
      <div class="loader-box">
        <i class="pi pi-spin pi-spinner"></i>
        <span>Running AI Forecast...</span>
      </div>
    </div>
  </div>
</template>


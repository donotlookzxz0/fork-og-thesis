<script setup>
import { ref, onMounted } from "vue"

const loading = ref(true)
const rows = ref([])
const globalRows = ref([])

const parseCategory = (metrics) => {
  return Object.keys(metrics).map(cat => ({
    category: cat,
    accuracy: metrics[cat].accuracy,
    macro_f1: metrics[cat].macro_f1,
    risk_mae: metrics[cat].risk_mae,
    total_items: metrics[cat].total_items
  }))
}

const objToRows = (obj) => {
  return Object.keys(obj).map(k => ({
    name: k,
    value: typeof obj[k] === "object"
      ? JSON.stringify(obj[k])
      : obj[k]
  }))
}

const load = async () => {
  const res = await fetch("https://api.pimart.software/metrics/stockout-risk")
  const data = await res.json()

  if (data.success) {
    rows.value = parseCategory(data.metrics)
    globalRows.value = objToRows(data.global_metrics)
  }

  loading.value = false
}

onMounted(load)
</script>

<template>
  <div style="padding:20px;color:white">
    <h2>Stockout Risk Metrics</h2>

    <div v-if="loading">Loading...</div>

    <div v-else>
      <h3>Global Metrics</h3>
      <pre>{{ globalRows }}</pre>

      <h3>By Category</h3>
      <table border="1">
        <tr>
          <th>Category</th>
          <th>Accuracy</th>
          <th>Macro F1</th>
          <th>Risk MAE</th>
          <th>Total</th>
        </tr>

        <tr v-for="r in rows" :key="r.category">
          <td>{{ r.category }}</td>
          <td>{{ r.accuracy }}</td>
          <td>{{ r.macro_f1 }}</td>
          <td>{{ r.risk_mae }}</td>
          <td>{{ r.total_items }}</td>
        </tr>
      </table>
    </div>
  </div>
</template>
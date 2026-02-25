<script setup>
import { ref, onMounted } from "vue"

const loading = ref(true)
const rows = ref([])

/* 🔥 Convert your API structure into table rows */
const parseMetrics = (metrics) => {
  const list = []

  Object.keys(metrics).forEach(category => {
    const ranges = metrics[category]

    Object.keys(ranges).forEach(range => {
      const m = ranges[range]

      list.push({
        category,
        range,
        mae: m.mae.toFixed(2),
        rmse: m.rmse.toFixed(2),
        mape: m.mape.toFixed(2)
      })
    })
  })

  return list
}

const load = async () => {
  console.log("🚀 calling direct API")

  try {
    const res = await fetch("https://api.pimart.software/metrics/forecast")
    const data = await res.json()

    console.log("DATA:", data)

    if (data.success) {
      rows.value = parseMetrics(data.metrics)
    }
  } catch (err) {
    console.error("ERROR:", err)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div style="padding:20px;color:white">

    <h2>Metrics (Direct API Test)</h2>

    <div v-if="loading">
      Loading...
    </div>

    <table v-else border="1" cellpadding="6">
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
          <td>{{ r.mae }}</td>
          <td>{{ r.rmse }}</td>
          <td>{{ r.mape }}</td>
        </tr>
      </tbody>
    </table>

  </div>
</template>
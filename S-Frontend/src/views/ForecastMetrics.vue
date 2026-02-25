<script setup>
import { ref, onMounted } from "vue"

const loading = ref(true)
const rows = ref([])

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
  const res = await fetch("https://api.pimart.software/metrics/forecast")
  const data = await res.json()

  if (data.success) {
    rows.value = parseMetrics(data.metrics)
  }

  loading.value = false
}

onMounted(load)
</script>

<template>
  <div style="padding:20px;color:white">
    <h2>Forecast Metrics</h2>

    <div v-if="loading">Loading...</div>

    <table v-else border="1" cellpadding="6">
      <tr>
        <th>Category</th>
        <th>Range</th>
        <th>MAE</th>
        <th>RMSE</th>
        <th>MAPE</th>
      </tr>

      <tr v-for="r in rows" :key="r.category+r.range">
        <td>{{ r.category }}</td>
        <td>{{ r.range }}</td>
        <td>{{ r.mae }}</td>
        <td>{{ r.rmse }}</td>
        <td>{{ r.mape }}</td>
      </tr>
    </table>
  </div>
</template>
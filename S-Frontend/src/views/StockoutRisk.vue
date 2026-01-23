<script setup>
import { ref, onMounted } from "vue"
import api from "../services/api"

const risks = ref([])

const load = async () => {
  risks.value = (await api.get("/ml/stockout-risk")).data
}

const run = async () => {
  await api.post("/ml/stockout-risk")
  load()
}

onMounted(load)
</script>

<template>
  <div class="page">
    <h1>Stockout Risk</h1>
    <button @click="run">Run Model</button>

    <table>
      <thead>
        <tr>
          <th>Item</th>
          <th>Category</th>
          <th>Stock</th>
          <th>Risk</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="r in risks" :key="r.item_id">
          <td>{{ r.item_name }}</td>
          <td>{{ r.category }}</td>
          <td>{{ r.current_stock }}</td>
          <td>{{ r.risk_level }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 20px;
}
th, td {
  border: 1px solid #333;
  padding: 10px;
}
</style>

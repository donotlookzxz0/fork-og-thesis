<script setup>
import { ref, onMounted, computed } from "vue"
import api from "../services/api"

const items = ref([])

const load = async () => {
  items.value = (await api.get("/ml/item-movement-forecast")).data
}

const run = async () => {
  await api.post("/ml/item-movement-forecast")
  load()
}

// Fast → Medium → Slow order
const movementOrder = {
  Fast: 1,
  Medium: 2,
  Slow: 3
}

const sortedItems = computed(() => {
  return [...items.value].sort(
    (a, b) =>
      (movementOrder[a.movement_class] || 99) -
      (movementOrder[b.movement_class] || 99)
  )
})

onMounted(load)
</script>

<template>
  <div class="page">
    <h1>Item Movement</h1>
    <button @click="run">Run Model</button>

    <table>
      <thead>
        <tr>
          <th>Item</th>
          <th>Category</th>
          <th>Avg Daily Sales</th>
          <th>Movement</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="i in sortedItems" :key="i.item_id">
          <td>{{ i.item_name }}</td>
          <td>{{ i.category }}</td>
          <td>{{ i.avg_daily_sales.toFixed(2) }}</td>
          <td>{{ i.movement_class }}</td>
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
th,
td {
  border: 1px solid #333;
  padding: 10px;
}
</style>

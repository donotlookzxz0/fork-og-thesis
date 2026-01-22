<script setup>
import { ref, onMounted, computed } from "vue"
import api from "../services/api"

/* ✅ FIX 1: remove trailing slash */
const getItems = () => api.get("items")
const createItem = (data) => api.post("items", data)
const updateItem = (id, data) => api.put(`items/${id}`, data)
const deleteItem = (id) => api.delete(`items/${id}`)

const categories = [
  "Fruits","Vegetables","Meat","Seafood","Dairy","Beverages","Snacks","Bakery",
  "Frozen","Canned Goods","Condiments","Dry Goods","Grains & Pasta",
  "Spices & Seasonings","Breakfast & Cereal","Personal Care","Household",
  "Baby Products","Pet Supplies","Health & Wellness","Cleaning Supplies"
]

const items = ref([])
const search = ref("")
const editMode = ref(false)
const currentId = ref(null)

const form = ref({
  name: "",
  quantity: 0,
  category: "",
  price: 0,
  barcode: ""
})

const filteredItems = computed(() => {
  if (!search.value) return items.value
  const q = search.value.toLowerCase()
  return items.value.filter(
    i => i.name.toLowerCase().includes(q) || i.barcode.includes(q)
  )
})

const fetchItems = async () => {
  const res = await getItems()
  console.log("ITEMS:", res.data) // keep for debugging
  items.value = res.data
}

const submitForm = async () => {
  if (editMode.value) {
    await updateItem(currentId.value, form.value)
  } else {
    await createItem(form.value)
  }
  resetForm()
  fetchItems()
}

const editItem = (item) => {
  editMode.value = true
  currentId.value = item.id
  form.value = { ...item }
}

const removeItem = async (id) => {
  if (!confirm("Delete this item?")) return
  await deleteItem(id)
  fetchItems()
}

const resetForm = () => {
  editMode.value = false
  currentId.value = null
  form.value = { name:"", quantity:0, category:"", price:0, barcode:"" }
}

onMounted(fetchItems)
</script>

<template>
  <div class="inventory">
    <h1><i class="pi pi-box"></i> Inventory</h1>

    <div class="search-wrapper">
      <i class="pi pi-search"></i>
      <input v-model="search" placeholder="Search by name or barcode" />
    </div>

    <form @submit.prevent="submitForm" class="form">
      <div class="input-icon">
        <i class="pi pi-tag"></i>
        <input v-model="form.name" placeholder="Name" />
      </div>

      <div class="input-icon">
        <i class="pi pi-sort-numeric-up"></i>
        <input v-model.number="form.quantity" type="number" placeholder="Quantity" />
      </div>

      <div class="input-icon">
        <i class="pi pi-list"></i>
        <select v-model="form.category">
          <option value="" disabled>Select category</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
      </div>

      <div class="input-icon">
        <i class="pi pi-dollar"></i>
        <input v-model.number="form.price" type="number" placeholder="Price" />
      </div>

      <div class="input-icon">
        <i class="pi pi-barcode"></i>
        <input v-model="form.barcode" placeholder="Barcode" />
      </div>

      <button type="submit" class="btn primary">
        <i :class="editMode ? 'pi pi-save' : 'pi pi-plus'"></i>
        {{ editMode ? "Update" : "Add" }}
      </button>

      <button v-if="editMode" type="button" class="btn secondary" @click="resetForm">
        <i class="pi pi-times"></i> Cancel
      </button>
    </form>

    <table>
      <thead>
        <tr>
          <th>Name</th>
          <th>Qty</th>
          <th>Category</th>
          <th>Price</th>
          <th>Barcode</th>
          <th>Actions</th>
        </tr>
      </thead>

      <!-- ✅ FIX 2: prevent empty/phantom rows -->
      <tbody v-if="filteredItems.length">
        <tr v-for="item in filteredItems" :key="item.id">
          <td>{{ item.name }}</td>
          <td>{{ item.quantity }}</td>
          <td>{{ item.category }}</td>
          <td>{{ item.price }}</td>
          <td>{{ item.barcode }}</td>
          <td class="actions">
            <button class="icon-btn edit" @click="editItem(item)">
              <i class="pi pi-pencil"></i>
            </button>
            <button class="icon-btn delete" @click="removeItem(item.id)">
              <i class="pi pi-trash"></i>
            </button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.inventory {
  padding: 20px;
}

h1 i {
  margin-right: 8px;
}

.search-wrapper {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 18px;
}

.search-wrapper input {
  width: 360px;
  height: 48px;
  font-size: 1rem;
  padding: 0 12px;
}

.form {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
}

.input-icon {
  position: relative;
}

.input-icon i {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: #888;
}

.input-icon input,
.input-icon select {
  height: 48px;
  min-width: 220px;
  padding-left: 38px;
  font-size: 1rem;
}

.btn {
  height: 48px;
  padding: 0 18px;
  font-size: 1rem;
  cursor: pointer;
}

.btn i {
  margin-right: 6px;
}

.primary {
  background: #3ddc97;
  border: none;
}

.secondary {
  background: #444;
  color: #fff;
  border: none;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th,
td {
  border: 1px solid #ddd;
  padding: 10px;
}

.actions {
  display: flex;
  gap: 8px;
}

.icon-btn {
  border: none;
  padding: 6px 10px;
  cursor: pointer;
}

.edit {
  background: #2196f3;
  color: white;
}

.delete {
  background: #e53935;
  color: white;
}
</style>

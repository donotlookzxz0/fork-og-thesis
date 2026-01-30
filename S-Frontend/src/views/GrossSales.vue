<template>
  <div class="p-4">

    <Toast position="top-center" />

    <Card>
      <template #title>
        Gross Sales Overview
      </template>

      <template #content>

        <!-- Filters -->
        <div class="grid mb-4">
          <div class="col-12 md:col-3">
            <label class="block mb-2">Start Date</label>
            <Calendar v-model="startDate" showIcon class="w-full" />
          </div>

          <div class="col-12 md:col-3">
            <label class="block mb-2">End Date</label>
            <Calendar v-model="endDate" showIcon class="w-full" />
          </div>

          <div class="col-12 md:col-3 flex align-items-end">
            <Button
              label="Load Sales"
              icon="pi pi-search"
              class="w-full"
              :loading="loading"
              @click="loadSales"
            />
          </div>
        </div>

        <!-- Summary Cards -->
        <div class="grid mb-4">
          <div class="col-12 md:col-4">
            <Card>
              <template #content>
                <div class="text-xl font-bold">Total Gross Sales</div>
                <div class="text-3xl mt-2">
                  {{ peso(totalGross) }}
                </div>
              </template>
            </Card>
          </div>

          <div class="col-12 md:col-4">
            <Card>
              <template #content>
                <div class="text-xl font-bold">Transactions</div>
                <div class="text-3xl mt-2">
                  {{ rows.length }}
                </div>
              </template>
            </Card>
          </div>

          <div class="col-12 md:col-4">
            <Card>
              <template #content>
                <div class="text-xl font-bold">Average Sale</div>
                <div class="text-3xl mt-2">
                  {{ peso(avgSale) }}
                </div>
              </template>
            </Card>
          </div>
        </div>

        <!-- Table -->
        <DataTable
          :value="rows"
          paginator
          :rows="10"
          stripedRows
          responsiveLayout="scroll"
          :loading="loading"
        >
          <Column field="date" header="Date" />
          <Column field="transaction_id" header="Transaction ID" />
          <Column field="items_count" header="Items" />
          <Column header="Gross Amount">
            <template #body="slotProps">
              {{ peso(slotProps.data.gross) }}
            </template>
          </Column>
        </DataTable>

      </template>
    </Card>
  </div>
</template>

<script setup>
import { ref, computed } from "vue"
import api from "../services/api"

/* PrimeVue */
import Card from "primevue/card"
import Button from "primevue/button"
import Calendar from "primevue/calendar"
import DataTable from "primevue/datatable"
import Column from "primevue/column"
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"

const toast = useToast()

/* ---------------- STATE ---------------- */

const startDate = ref(null)
const endDate = ref(null)
const rows = ref([])
const loading = ref(false)

/* ---------------- COMPUTED ---------------- */

const totalGross = computed(() =>
  rows.value.reduce((s, r) => s + r.gross, 0)
)

const avgSale = computed(() =>
  rows.value.length ? totalGross.value / rows.value.length : 0
)

/* ---------------- HELPERS ---------------- */

function peso(v) {
  return `₱${Number(v || 0).toFixed(2)}`
}

function inRange(dateStr) {
  if (!startDate.value && !endDate.value) return true
  const d = new Date(dateStr)
  if (startDate.value && d < startDate.value) return false
  if (endDate.value && d > endDate.value) return false
  return true
}

/* ---------------- MAIN ---------------- */

async function loadSales() {
  loading.value = true

  try {
    const res = await api.get("/sales")

    const mapped = res.data
      .filter(t => inRange(t.date))
      .map(t => {
        let gross = 0
        let items = 0

        for (const i of t.items) {
          gross += i.quantity * i.price_at_sale
          items += i.quantity
        }

        return {
          transaction_id: t.transaction_id,
          date: new Date(t.date).toLocaleString(),
          gross,
          items_count: items
        }
      })

    rows.value = mapped

    toast.add({
      severity: "success",
      summary: "Loaded",
      detail: "Sales computed from transactions",
      life: 2500
    })

  } catch (err) {
    console.error(err)

    toast.add({
      severity: "error",
      summary: "Load Failed",
      detail: "Could not load sales",
      life: 3000
    })
  }

  loading.value = false
}
</script>

<style scoped>
.text-3xl {
  font-size: 1.75rem;
  font-weight: bold;
}
</style>

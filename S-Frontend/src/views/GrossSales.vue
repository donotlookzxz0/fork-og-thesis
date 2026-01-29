<template>
  <div class="p-4">
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
                  {{ currency(totalGross) }}
                </div>
              </template>
            </Card>
          </div>

          <div class="col-12 md:col-4">
            <Card>
              <template #content>
                <div class="text-xl font-bold">Transactions</div>
                <div class="text-3xl mt-2">
                  {{ sales.length }}
                </div>
              </template>
            </Card>
          </div>

          <div class="col-12 md:col-4">
            <Card>
              <template #content>
                <div class="text-xl font-bold">Average Sale</div>
                <div class="text-3xl mt-2">
                  {{ currency(avgSale) }}
                </div>
              </template>
            </Card>
          </div>
        </div>

        <!-- Sales Table -->
        <DataTable
          :value="sales"
          paginator
          :rows="10"
          stripedRows
          responsiveLayout="scroll"
        >
          <Column field="date" header="Date" />
          <Column field="transaction_id" header="Transaction ID" />
          <Column field="items" header="Items" />
          <Column field="gross" header="Gross Amount">
            <template #body="slotProps">
              {{ currency(slotProps.data.gross) }}
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

/* ---------------- STATE ---------------- */

const startDate = ref(null)
const endDate = ref(null)
const sales = ref([])

/* ---------------- COMPUTED ---------------- */

const totalGross = computed(() =>
  sales.value.reduce((sum, s) => sum + Number(s.gross || 0), 0)
)

const avgSale = computed(() =>
  sales.value.length
    ? totalGross.value / sales.value.length
    : 0
)

/* ---------------- METHODS ---------------- */

function currency(v) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD"
  }).format(v)
}

async function loadSales() {
  try {
    const res = await api.get("/sales/gross", {
      params: {
        start: startDate.value,
        end: endDate.value
      }
    })

    // Expected response format example:
    // [
    //   { date, transaction_id, items, gross }
    // ]

    sales.value = res.data
  } catch (err) {
    console.error("Failed to load gross sales:", err)

    // fallback mock so page still works during backend dev
    sales.value = [
      { date: "2026-01-01", transaction_id: "TX1001", items: 4, gross: 120.50 },
      { date: "2026-01-01", transaction_id: "TX1002", items: 2, gross: 59.99 },
      { date: "2026-01-02", transaction_id: "TX1003", items: 6, gross: 210.00 }
    ]
  }
}
</script>

<style scoped>
.text-3xl {
  font-size: 1.75rem;
  font-weight: bold;
}
</style>

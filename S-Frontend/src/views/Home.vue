<script setup>
import { useRouter } from "vue-router"
import Card from "primevue/card"
import Button from "primevue/button"

const router = useRouter()
const go = (path) => router.push(path)
</script>

<template>
  <div class="home">
    <h1 class="title">Welcome, G friends</h1>
    <p class="subtitle">Pi-Mart Smart Self-Service Dashboard</p>

    <!-- STYLISH BENTO GRID -->
    <div class="bento">
      <!-- POS -->
      <Card class="bento-card pos gradient-orange">
        <template #title>POS</template>
        <template #content>
          Fast cashier operations with barcode scanning.
        </template>
        <template #footer>
          <Button label="Open POS" icon="pi pi-shopping-cart" @click="go('/pos')" />
        </template>
      </Card>

      <!-- INVENTORY -->
      <Card class="bento-card small gradient-blue">
        <template #title>Inventory</template>
        <template #content>
          Stock levels and item management.
        </template>
        <template #footer>
          <Button label="View" icon="pi pi-box" @click="go('/inventory')" />
        </template>
      </Card>

      <!-- TRANSACTIONS -->
      <Card class="bento-card small gradient-purple">
        <template #title>Transactions</template>
        <template #content>
          Completed sales records.
        </template>
        <template #footer>
          <Button label="History" icon="pi pi-receipt" @click="go('/transactions')" />
        </template>
      </Card>

      <!-- DEMAND FORECAST -->
      <Card class="bento-card wide gradient-green">
        <template #title>Demand Forecast</template>
        <template #content>
          Time-series demand prediction by category.
        </template>
        <template #footer>
          <Button
            icon="pi pi-chart-line"
            label="Analyze"
            @click="go('/analytics/demand')"
          />
        </template>
      </Card>

      <!-- ITEM MOVEMENT -->
      <Card class="bento-card small gradient-cyan">
        <template #title>Item Movement</template>
        <template #content>
          Fast, medium, slow movers.
        </template>
        <template #footer>
          <Button
            icon="pi pi-sort-amount-up"
            label="View"
            @click="go('/analytics/movement')"
          />
        </template>
      </Card>

      <!-- STOCKOUT -->
      <Card class="bento-card wide gradient-red">
        <template #title>Stockout Risk</template>
        <template #content>
          Items at risk of running out.
        </template>
        <template #footer>
          <Button
            icon="pi pi-exclamation-triangle"
            severity="danger"
            label="Inspect"
            @click="go('/analytics/stockout')"
          />
        </template>
      </Card>
    </div>

    <!-- FOOTER -->
    <footer class="footer">
      © 2026 Holy Angel University — Pi-Mart Smart Self-Service Grocery System.  
      Developed for academic and research purposes.
    </footer>
  </div>
</template>

<style scoped>
/* PAGE */
.home {
  padding: 24px;
  min-height: calc(100vh - 40px);
  display: flex;
  flex-direction: column;

  /* INHERIT GLOBAL BACKGROUND */
  background: transparent;
}


/* HEADER */
.title {
  color: #ffffff;
  font-size: 1.6rem;
}

.subtitle {
  color: #9ca3af;
  margin-bottom: 20px;
}

/* BENTO GRID */
.bento {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-auto-rows: 160px;
  gap: 16px;
  flex: 1;
}

/* CARD BASE */
.bento-card {
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(8px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.bento-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
}

/* SIZES */
.pos {
  grid-column: span 2;
  grid-row: span 2;
}

.wide {
  grid-column: span 2;
}

.small {
  grid-column: span 1;
}

/* GRADIENTS */
.gradient-orange {
  background: linear-gradient(135deg, #ff7a18, #ffb347);
}

.gradient-blue {
  background: linear-gradient(135deg, #2563eb, #60a5fa);
}

.gradient-purple {
  background: linear-gradient(135deg, #7c3aed, #a78bfa);
}

.gradient-green {
  background: linear-gradient(135deg, #059669, #34d399);
}

.gradient-cyan {
  background: linear-gradient(135deg, #0891b2, #67e8f9);
}

.gradient-red {
  background: linear-gradient(135deg, #dc2626, #f87171);
}

/* PRIMEVUE TEXT */
:deep(.p-card-title) {
  font-size: 1.1rem;
  font-weight: 600;
}

:deep(.p-card-content) {
  color: rgba(255, 255, 255, 0.85);
}

/* FOOTER */
.footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid #2a2a2a;
  text-align: center;
  font-size: 0.8rem;
  color: #9ca3af;
}

/* RESPONSIVE */
@media (max-width: 1024px) {
  .bento {
    grid-template-columns: repeat(2, 1fr);
  }

  .pos,
  .wide {
    grid-column: span 2;
  }
}

@media (max-width: 640px) {
  .bento {
    grid-template-columns: 1fr;
  }

  .bento-card {
    grid-column: span 1 !important;
    grid-row: span 1 !important;
  }
}
</style>

<script setup>
import { useRouter } from "vue-router"
import Card from "primevue/card"
import Button from "primevue/button"

const router = useRouter()
const go = (path) => router.push(path)
</script>

<template>
  <div class="home">
    <h1 class="title">Welcome to G-Friends</h1>
    <p class="subtitle">Pi-Mart Smart Self-Service Dashboard</p>

    <!-- BENTO GRID -->
    <div class="bento">
      <!-- POS (BIG LEFT) -->
      <Card class="bento-card pos gradient-orange">
        <template #title>POS</template>
        <template #content>
          Fast cashier operations with barcode scanning and live totals.
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
          Time-series demand prediction by product category.
        </template>
        <template #footer>
          <Button icon="pi pi-chart-line" label="Analyze" @click="go('/analytics/demand')" />
        </template>
      </Card>

      <!-- ITEM MOVEMENT -->
      <Card class="bento-card small gradient-cyan">
        <template #title>Item Movement</template>
        <template #content>
          Fast, medium, slow movers.
        </template>
        <template #footer>
          <Button icon="pi pi-sort-amount-up" label="View" @click="go('/analytics/movement')" />
        </template>
      </Card>

      <!-- PAYMENT -->
      <Card class="bento-card small gradient-indigo">
        <template #title>Payment</template>
        <template #content>
          Process digital and cash payments.
        </template>
        <template #footer>
          <Button icon="pi pi-credit-card" label="Open" @click="go('/payment')" />
        </template>
      </Card>

      <!-- WALLET TOP-UP -->
      <Card class="bento-card small gradient-teal">
        <template #title>Wallet Top-Up</template>
        <template #content>
          Add balance to customer wallets.
        </template>
        <template #footer>
          <Button icon="pi pi-wallet" label="Top-Up" @click="go('/wallet/top-up')" />
        </template>
      </Card>

      <!-- RECOMMENDATIONS -->
      <Card class="bento-card small gradient-gold">
        <template #title>Recommendations</template>
        <template #content>
          Smart restock and pricing suggestions.
        </template>
        <template #footer>
          <Button icon="pi pi-star" label="Open" @click="go('/analytics/recommendations')" />
        </template>
      </Card>

      <!-- STOCKOUT RISK (FULL WIDTH BOTTOM) -->
      <Card class="bento-card stockout gradient-red">
        <template #title>Stockout Risk</template>
        <template #content>
          Products likely to run out soon.
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
  grid-auto-rows: 170px;
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

/* 🔴 STOCKOUT FULL WIDTH */
.stockout {
  grid-column: span 4;
}

/* GRADIENTS */
.gradient-orange { background: linear-gradient(135deg, #ff7a18, #ffb347); }
.gradient-blue   { background: linear-gradient(135deg, #3b82f6, #60a5fa); }
.gradient-purple{ background: linear-gradient(135deg, #8b5cf6, #a78bfa); }
.gradient-green { background: linear-gradient(135deg, #10b981, #34d399); }
.gradient-cyan  { background: linear-gradient(135deg, #22c1dc, #67e8f9); }
.gradient-red   { background: linear-gradient(135deg, #ef4444, #f87171); }
.gradient-indigo{ background: linear-gradient(135deg, #4338ca, #818cf8); }
.gradient-teal  { background: linear-gradient(135deg, #0f766e, #5eead4); }
.gradient-gold  { background: linear-gradient(135deg, #ca8a04, #fde047); }

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
  .wide,
  .stockout {
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

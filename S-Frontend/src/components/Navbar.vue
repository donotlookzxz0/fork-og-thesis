<script setup>
import { ref } from "vue"
import { useRouter, useRoute } from "vue-router"
import api from "../services/api"

const router = useRouter()
const route = useRoute()

const isOpen = ref(false)
const analyticsOpen = ref(false)

const tabs = [
  { label: "Home", path: "/", icon: "pi pi-home" },
  { label: "Inventory", path: "/inventory", icon: "pi pi-box" },
  { label: "POS", path: "/pos", icon: "pi pi-shopping-cart" },
  { label: "Payment", path: "/payment", icon: "pi pi-credit-card" },
  { label: "Transactions", path: "/transactions", icon: "pi pi-receipt" },
  { label: "Wallet Top-Up", path: "/wallet/top-up", icon: "pi pi-wallet" },
  { label: "Gross Sales", path: "/analytics/gross-sales", icon: "pi pi-dollar" }
]

const analyticsTabs = [
  { label: "Demand Forecast", path: "/analytics/demand", icon: "pi pi-chart-line" },
  { label: "Item Movement", path: "/analytics/movement", icon: "pi pi-sort-amount-up" },
  { label: "Stockout Risk", path: "/analytics/stockout", icon: "pi pi-exclamation-triangle" },
  { label: "Recommendations", path: "/analytics/recommendations", icon: "pi pi-star" }
]

const navigate = (path) => {
  isOpen.value = false
  analyticsOpen.value = false
  router.push(path)
}

const handleLogout = async () => {
  try {
    await api.post("/users/logout")
  } finally {
    isOpen.value = false
    analyticsOpen.value = false
    router.push("/login")
  }
}
</script>

<template>
  <nav class="folder-nav">
    <button class="hamburger mobile" @click="isOpen = !isOpen">
      <i class="pi pi-bars" />
    </button>

    <div class="tabs desktop">
      <button
        v-for="tab in tabs"
        :key="tab.path"
        class="tab"
        :class="{ active: route.path === tab.path }"
        @click="navigate(tab.path)"
      >
        <i :class="tab.icon" />
        <span>{{ tab.label }}</span>
      </button>

      <div
        class="tab dropdown"
        :class="{
          active:
            route.path.startsWith('/analytics') &&
            route.path !== '/analytics/gross-sales'
        }"
      >
        <i class="pi pi-chart-line" />
        <span>Analytics</span>

        <div class="dropdown-menu">
          <button
            v-for="a in analyticsTabs"
            :key="a.path"
            class="dropdown-item"
            @click="navigate(a.path)"
          >
            <i :class="a.icon" />
            <span>{{ a.label }}</span>
          </button>
        </div>
      </div>
    </div>

    <div class="spacer"></div>

    <button class="logout desktop" @click="handleLogout">
      <i class="pi pi-sign-out" />
    </button>

    <div v-if="isOpen" class="backdrop" @click="isOpen = false"></div>

    <div v-if="isOpen" class="mobile-menu">
      <button
        v-for="tab in tabs"
        :key="tab.path"
        class="mobile-item"
        :class="{ active: route.path === tab.path }"
        @click="navigate(tab.path)"
      >
        <i :class="tab.icon" />
        <span>{{ tab.label }}</span>
      </button>

      <button class="mobile-item" @click="analyticsOpen = !analyticsOpen">
        <i class="pi pi-chart-line" />
        <span>Analytics</span>
        <i class="pi pi-chevron-down" :class="{ rotate: analyticsOpen }" />
      </button>

      <div v-if="analyticsOpen" class="mobile-submenu">
        <button
          v-for="a in analyticsTabs"
          :key="a.path"
          class="mobile-subitem"
          :class="{ active: route.path === a.path }"
          @click="navigate(a.path)"
        >
          <i :class="a.icon" />
          <span>{{ a.label }}</span>
        </button>
      </div>

      <button class="mobile-item logout-item" @click="handleLogout">
        <i class="pi pi-sign-out" />
        Logout
      </button>
    </div>
  </nav>
</template>

<style scoped>
.folder-nav {
  display: flex;
  align-items: flex-end;
  padding: 0 8px;
  height: 40px;
  background-color: transparent;
  position: relative;
  z-index: 101;
}

.folder-nav::after {
  display: none;
}

.tabs {
  display: flex;
  gap: 4px;
}

.tab {
  min-width: 96px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  background-color: #252525;
  color: #bdbdbd;
  border: 1px solid #2f2f2f;
  border-bottom: none;
  border-radius: 8px 8px 0 0;
  cursor: pointer;
  font-size: 0.8rem;
  position: relative;
  box-shadow: inset 0 -1px 0 rgba(255, 255, 255, 0.03);
}

.tab.active {
  background-color: #1e1e1e;
  color: #ffffff;
  border-color: #3a3a3a;
  transform: translateY(1px);
  z-index: 2;
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04),
    0 -1px 0 rgba(0, 0, 0, 0.4);
}

.dropdown {
  position: relative;
}

.dropdown::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 100%;
  height: 8px;
}

.dropdown-menu {
  position: absolute;
  top: 34px;
  left: 0;
  background-color: #1e1e1e;
  border: 1px solid #333;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  min-width: 200px;
  z-index: 200;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 0.15s ease;
}

.dropdown:hover .dropdown-menu,
.dropdown-menu:hover {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
}

.dropdown-item {
  padding: 10px 12px;
  display: flex;
  align-items: center;
  gap: 8px;
  background: transparent;
  color: #cfcfcf;
  border: none;
  cursor: pointer;
  text-align: left;
}

.dropdown-item:hover {
  background-color: #2a2a2a;
  color: #ffffff;
}

.spacer {
  flex: 1;
}

.logout {
  height: 35px;
  width: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: #ff6b6b;
  border: 1px solid #444;
  border-radius: 6px;
  cursor: pointer;
}

.hamburger {
  height: 32px;
  width: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: #cfcfcf;
  border: 1px solid #444;
  border-radius: 6px;
  cursor: pointer;
}

.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.45);
  z-index: 99;
}

.mobile-menu {
  position: fixed;
  top: 40px;
  left: 8px;
  width: 240px;
  background-color: #1e1e1e;
  border: 1px solid #333;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  z-index: 100;
}

.mobile-item,
.mobile-subitem {
  padding: 12px 14px;
  display: flex;
  align-items: center;
  gap: 10px;
  background: transparent;
  color: #cfcfcf;
  border: none;
  text-align: left;
  cursor: pointer;
}

.mobile-item:hover,
.mobile-subitem:hover {
  background-color: #2a2a2a;
}

.mobile-item.active,
.mobile-subitem.active {
  background-color: #242424;
  color: #ffffff;
}

.mobile-submenu {
  padding-left: 12px;
}

.logout-item {
  color: #ff6b6b;
}

.rotate {
  margin-left: auto;
  transform: rotate(180deg);
}

.desktop {
  display: flex;
}

.mobile {
  display: none;
}

@media (max-width: 768px) {
  .folder-nav {
    align-items: center;
  }

  .desktop {
    display: none;
  }

  .mobile {
    display: flex;
  }
}
</style>

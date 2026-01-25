<script setup>
import { ref, onMounted, computed } from "vue"
import { useRouter } from "vue-router"
import api from "../services/api"

// PrimeVue Toast + Confirm
import Toast from "primevue/toast"
import { useToast } from "primevue/usetoast"
import ConfirmDialog from "primevue/confirmdialog"
import { useConfirm } from "primevue/useconfirm"

const toast = useToast()
const confirm = useConfirm()
const router = useRouter()

const username = ref("")
const selectedUser = ref(null)
const amount = ref("")
const loading = ref(false)
const users = ref([])
const balance = ref(0)

/* =========================
   FETCH USERS (ADMIN ONLY)
========================= */
const fetchUsers = async () => {
  try {
    const res = await api.get("/users")
    users.value = res.data
  } catch (err) {
    console.error("Users fetch error:", err)

    if (err.response?.status === 401) {
      toast.add({
        severity: "error",
        summary: "Session Expired",
        detail: "Please login again",
        life: 3000,
      })
      setTimeout(() => router.push("/login"), 800)
    } else {
      toast.add({
        severity: "error",
        summary: "Error",
        detail: err.response?.data?.error || "Failed to load users",
        life: 3000,
      })
    }
  }
}

onMounted(fetchUsers)

/* =========================
   FILTER USERS
========================= */
const filteredUsers = computed(() => {
  const q = username.value.trim().toLowerCase()
  if (!q || selectedUser.value) return []
  return users.value.filter(u =>
    u.username.toLowerCase().includes(q)
  )
})

/* =========================
   SELECT USER
========================= */
const selectUser = async (u) => {
  try {
    selectedUser.value = u
    username.value = u.username
    balance.value = 0

    const res = await api.get(
      `/payment/admin/wallet/balance/${u.id}`
    )

    balance.value = res.data.balance

  } catch (err) {
    console.error("Balance fetch error:", err)

    if (err.response?.status === 401) {
      toast.add({
        severity: "error",
        summary: "Session Expired",
        detail: "Please login again",
        life: 3000,
      })
      setTimeout(() => router.push("/login"), 800)
    } else {
      toast.add({
        severity: "error",
        summary: "Error",
        detail: err.response?.data?.error || "Failed to fetch balance",
        life: 3000,
      })
    }
  }
}

/* =========================
   RESET FORM
========================= */
const resetForm = () => {
  username.value = ""
  selectedUser.value = null
  amount.value = ""
  balance.value = 0
}

/* =========================
   CASH IN  (MODERN CONFIRM)
========================= */
const cashIn = async () => {
  if (!selectedUser.value) {
    toast.add({
      severity: "warn",
      summary: "No User",
      detail: "Please select a user first",
      life: 2000,
    })
    return
  }

  if (!amount.value || Number(amount.value) <= 0) {
    toast.add({
      severity: "warn",
      summary: "Invalid Amount",
      detail: "Enter a valid cash-in amount",
      life: 2000,
    })
    return
  }

  // 🔥 REPLACED BROWSER CONFIRM WITH PRIMEVUE CONFIRM DIALOG
  confirm.require({
    header: "Confirm Cash-In",
    message: `Cash in ₱${amount.value} to ${selectedUser.value.username}?`,
    icon: "pi pi-wallet",
    acceptLabel: "Confirm",
    rejectLabel: "Cancel",
    acceptClass: "p-button-success",
    rejectClass: "p-button-danger",
    position: "top",

    accept: async () => {
      loading.value = true

      try {
        const res = await api.post("/payment/admin/wallet/topup", {
          user_id: selectedUser.value.id,
          amount: Number(amount.value),
        })

        // ✅ SUCCESS POPUP (CENTER STYLE)
        toast.add({
          severity: "success",
          summary: "Cash-In Successful",
          detail: `New Balance: ₱${res.data.new_balance}`,
          life: 3000,
        })

        /* ✅ AUTO RESET AFTER SUCCESS */
        resetForm()

      } catch (err) {
        console.error("Topup error:", err)

        if (err.response?.status === 401) {
          toast.add({
            severity: "error",
            summary: "Session Expired",
            detail: "Please login again",
            life: 3000,
          })
          setTimeout(() => router.push("/login"), 800)
        } else {
          toast.add({
            severity: "error",
            summary: "Cash-In Failed",
            detail: err.response?.data?.error || "Cash-in failed",
            life: 3000,
          })
        }

      } finally {
        loading.value = false
      }
    }
  })
}
</script>

<template>
  <div class="wallet-wrapper">

    <!-- 🔔 TOAST POPUPS (CENTER STYLE) -->
    <Toast position="top-center" />

    <!-- 🔐 CONFIRM DIALOG -->
    <ConfirmDialog />

    <div class="wallet">
      <h1 class="title"><i class="pi pi-wallet"></i>Wallet Cash-In</h1>

      <div class="card">

        <!-- USER SEARCH -->
        <div class="input-icon">
          <i class="pi pi-user"></i>
          <input
            v-model="username"
            placeholder="Type username"
            @focus="selectedUser = null"
          />
        </div>

        <!-- SEARCH RESULTS -->
        <div v-if="filteredUsers.length" class="results">
          <div
            v-for="u in filteredUsers"
            :key="u.id"
            class="result-item"
            @click="selectUser(u)"
          >
            <div class="user-main">{{ u.username }}</div>
            <div class="user-sub">ID: {{ u.id }}</div>
          </div>
        </div>

        <!-- SELECTED USER INFO -->
        <div v-if="selectedUser" class="user-info">
          <i class="pi pi-check-circle"></i>
          <div>
            <strong>{{ selectedUser.username }}</strong>
            <div class="muted">
              ID: {{ selectedUser.id }} · Balance: ₱{{ balance }}
            </div>
          </div>
        </div>

        <!-- AMOUNT -->
        <div class="input-icon amount">
          <i class="pi pi-dollar"></i>
          <input
            v-model.number="amount"
            type="number"
            min="1"
            placeholder="Cash-in amount"
          />
        </div>

        <!-- CASH IN BUTTON -->
        <button
          class="btn cashin w-full"
          :disabled="loading || !selectedUser"
          @click="cashIn"
        >
          <i class="pi pi-arrow-down"></i>
          {{ loading ? "Processing..." : "Cash In Wallet" }}
        </button>

        <p class="hint">
          Select a user, review balance, then top up.
        </p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.wallet-wrapper {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 40px);
}

.wallet {
  width: 100%;
  max-width: 460px;
  padding: 20px;
}

.card {
  border: 1px solid #ddd;
  padding: 20px;
  background: #fff;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.input-icon {
  position: relative;
}

.input-icon i {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: #555;
}

.input-icon input {
  height: 48px;
  width: 100%;
  padding-left: 38px;
  font-size: 1rem;
}

.input-icon.amount input {
  color: #1d4ed8;
  font-weight: 600;
}

.results {
  border: 1px solid #ddd;
}

.result-item {
  padding: 10px;
  cursor: pointer;
}

.result-item:hover {
  background: #f3f4f6;
}

.user-main {
  color: #000;
  font-weight: 600;
}

.user-sub {
  font-size: 13px;
  color: #444;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px;
  border: 1px solid #ddd;
  background: #f9f9f9;
}

.user-info i {
  color: #16a34a;
}

.btn {
  height: 48px;
  border: none;
  cursor: pointer;
}

.cashin {
  background: #16a34a;
  color: white;
}

.cashin:disabled {
  background: #86efac;
}

.w-full {
  width: 100%;
}

.muted {
  font-size: 13px;
  color: #444;
}

.hint {
  font-size: 13px;
  color: #666;
  text-align: center;
}

/* Match title size with other pages */
.title {
  color: #ffffff;
  font-size: 1.8rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}
</style>

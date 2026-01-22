<script setup>
import { ref } from "vue"
import { useRouter } from "vue-router"
import api from "../services/api"

import InputText from "primevue/inputtext"
import Password from "primevue/password"
import Button from "primevue/button"

const router = useRouter()
const username = ref("")
const password = ref("")
const error = ref("")
const loading = ref(false)

const login = async () => {
  error.value = ""
  loading.value = true
  try {
    await api.post("/users/login", {
      username: username.value,
      password: password.value
    })
    router.push("/")
  } catch (err) {
    error.value = err.response?.data?.error || "Login failed"
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-wrapper">
    <div class="login-card">
      <div class="title">
        <i class="pi pi-lock"></i>
        <span>Admin Login</span>
      </div>

      <div class="form">
        <div class="field">
          <label>Username</label>
          <InputText
            v-model="username"
            class="auth-input"
          />
        </div>

        <div class="field">
          <label>Password</label>
          <Password
            v-model="password"
            toggleMask
            :feedback="false"
            class="auth-input"
          />
        </div>

        <small v-if="error" class="error">{{ error }}</small>

        <Button
          label="Login"
          icon="pi pi-arrow-right"
          class="login-btn"
          :loading="loading"
          @click="login"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.login-wrapper {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: radial-gradient(circle at top, #2a2a2a, #0f0f0f);
}

.login-card {
  width: 420px;
  background: #121417;
  border-radius: 18px;
  padding: 2rem 2.25rem;
  box-shadow: 0 30px 70px rgba(0, 0, 0, 0.7);
}

.title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 1.35rem;
  font-weight: 600;
  color: #ffffff;
  margin-bottom: 1.5rem;
}

.form {
  display: flex;
  flex-direction: column;
  gap: 1.1rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

label {
  font-size: 0.85rem;
  color: #cfcfcf;
}

/* ========= CRITICAL FIX ========= */
.auth-input {
  width: 100%;
}

/* InputText */
:deep(.auth-input.p-inputtext) {
  width: 100%;
  height: 52px;
}

/* Password wrapper */
:deep(.auth-input.p-password) {
  width: 100%;
}

/* Password input */
:deep(.auth-input .p-password-input) {
  width: 100%;
  height: 52px;
}

/* Shared appearance */
:deep(.auth-input.p-inputtext),
:deep(.auth-input .p-password-input) {
  padding: 0 1rem;
  border-radius: 12px;
  background: transparent;
  border: 1px solid #2f2f2f;
  color: #ffffff;
  box-sizing: border-box;
}

/* Focus glow */
:deep(.auth-input.p-inputtext:focus),
:deep(.auth-input .p-password-input:focus) {
  border-color: #3ddc97;
  box-shadow: 0 0 0 1px #3ddc97;
}

/* Eye icon alignment */
:deep(.p-password-toggle) {
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
}

.login-btn {
  margin-top: 0.5rem;
  height: 52px;
  border-radius: 12px;
  font-weight: 600;
  background: linear-gradient(135deg, #3ddc97, #2bcf88);
  border: none;
  color: #000;
}

.error {
  text-align: center;
  color: #ff6b6b;
  font-size: 0.85rem;
}
</style>

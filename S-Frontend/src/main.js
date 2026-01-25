import { createApp } from "vue"
import App from "./App.vue"
import router from "./router"

import PrimeVue from "primevue/config"
import Aura from "@primeuix/themes/aura"
import ToastService from "primevue/toastservice"
import ConfirmationService from "primevue/confirmationservice"

import "primeicons/primeicons.css"
import "./style.css"

createApp(App)
  .use(router)
  .use(PrimeVue, {
    theme: { preset: Aura }
  })
  .use(ToastService)           // 🔔 Toast popups
  .use(ConfirmationService)   // 🔐 Confirm dialogs
  .mount("#app")

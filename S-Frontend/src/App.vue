<script setup>
import { computed } from "vue"
import { useRoute } from "vue-router"
import Navbar from "./components/Navbar.vue"

const route = useRoute()

const showNavbar = computed(() => route.path !== "/login")
</script>

<template>
  <Navbar v-if="showNavbar" />
  <router-view />
</template>

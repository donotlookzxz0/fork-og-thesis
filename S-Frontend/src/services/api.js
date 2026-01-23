import axios from "axios";

const api = axios.create({
  baseURL: "/api",                 // 🔥 CRITICAL FIX
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

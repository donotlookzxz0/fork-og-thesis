import axios from "axios";

const api = axios.create({
  baseURL: "/api",          // ✅ NO TRAILING SLASH (IMPORTANT)
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

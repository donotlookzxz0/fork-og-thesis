import axios from "axios";

const api = axios.create({
  baseURL: "/api/",        // 🔥 FORCE VERCEL PROXY
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

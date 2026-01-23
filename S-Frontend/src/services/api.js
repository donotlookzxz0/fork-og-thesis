import axios from "axios";

const api = axios.create({
  baseURL: "/api",                 // 🔥 IMPORTANT — USE VERCEL REWRITE PIPE
  withCredentials: true,           // 🔥 REQUIRED FOR COOKIE AUTH
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

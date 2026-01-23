import axios from "axios";

const api = axios.create({
  baseURL: "https://api.pimart.software",
  withCredentials: true,          // 🔥 REQUIRED FOR COOKIE AUTH
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

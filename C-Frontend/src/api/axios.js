import axios from "axios";

const api = axios.create({
  baseURL: "https://api.pimart.software",   // 🔥 MUST BE HTTPS
  withCredentials: true,                   // 🔑 send JWT cookies
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

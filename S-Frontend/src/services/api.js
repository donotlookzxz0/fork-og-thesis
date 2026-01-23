import axios from "axios";

const api = axios.create({
  baseURL: "https://api.pimart.software",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;
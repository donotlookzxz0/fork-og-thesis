import axios from "axios";

const api = axios.create({
  baseURL: "https://api.pimart.software",
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;

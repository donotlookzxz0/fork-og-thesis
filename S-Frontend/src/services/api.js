import axios from "axios";

const api = axios.create({
  baseURL: "https://chippable-lawrence-seventhly.ngrok-free.dev",
  withCredentials: true,
  headers: {
    "Content-Type": "application/json",
  },
});

export default api;
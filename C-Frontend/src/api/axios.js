import axios from "axios";

const api = axios.create({
  baseURL: "https://api.pimart.software",
  withCredentials: true,
});

export default api;

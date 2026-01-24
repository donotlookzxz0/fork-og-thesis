import axios from "axios";

console.log("AXIOS BASE URL:", "https://api.pimart.software");

const api = axios.create({
  baseURL: "https://api.pimart.software",
  withCredentials: true,
});

export default api;

import axios from "axios";

console.log("AXIOS BASE URL:", "https://api.pimart.software");

const api = axios.create({
  baseURL: "http://127.0.0.1:5000",
  withCredentials: true,
});

export default api;

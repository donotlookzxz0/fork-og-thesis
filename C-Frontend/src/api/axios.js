// import axios from "axios";

// const api = axios.create({
//   baseURL: "http://127.0.0.1:5000", // ✅ Flask backend
//   withCredentials: true,
// });

// export default api;




//use below for live service


import axios from "axios";

const api = axios.create({
  baseURL: "/api",              // 🔥 SAME-SITE
  withCredentials: true,
});

export default api;

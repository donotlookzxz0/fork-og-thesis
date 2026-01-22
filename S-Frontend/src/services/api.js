
// import axios from "axios";

// const api = axios.create({
//   baseURL: "http://127.0.0.1:5000", // ✅ Flask API
//   withCredentials: true,
//   headers: {
//     "Content-Type": "application/json",
//   },
// });

// export default api;



//use below for live service

import axios from "axios";

  const api = axios.create({
    baseURL: "/api",               // 🔥 SAME-SITE via Vercel proxy
    withCredentials: true,
    headers: {
      "Content-Type": "application/json",
    },
  });

  export default api;

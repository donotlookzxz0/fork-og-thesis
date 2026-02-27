
// import axios from "axios";

// console.log("AXIOS BASE URL:", "https://api.pimart.software");

// const api = axios.create({
//   baseURL: "https://api.pimart.software",
//   withCredentials: true,
// });

// export default api;

import axios from "axios";

const api = axios.create({
  baseURL:
    process.env.NODE_ENV === "development"
      ? "http://127.0.0.1:5000"  // :small_blue_diamond: localhost backend for dev
      : "/api",                 // :small_blue_diamond: production uses Vercel rewrite
  withCredentials: true,
});

export default api;
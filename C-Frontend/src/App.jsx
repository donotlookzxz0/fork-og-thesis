import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import ProtectedRoute from "./components/ProtectedRoute";
import Header from "./components/navbar/header";
import Home from "./screens/landing/page";
import ScannerPage from "./screens/add-to-cart";
import ViewCart from "./screens/view-cart";
import Checkout from "./components/Checkout";
import BestSellers from "./components/BestSellers";
import Wallet from "./screens/Wallet"; 
import Login from "./screens/auth/Login";
import Register from "./screens/auth/Register";
import Success from "./screens/success/page";

import "./App.css";

function App() {
  const [cart, setCart] = React.useState(() => {
    try {
      return JSON.parse(localStorage.getItem("cart")) || [];
    } catch {
      return [];
    }
  });

  React.useEffect(() => {
    localStorage.setItem("cart", JSON.stringify(cart));
  }, [cart]);

   return (
    <Router>
      <div className="background">
      </div>
      
      <Routes>
        {/* Public */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Protected */}
        <Route
          path="/cart"
          element={
            <ProtectedRoute>
              <Header />
              <ViewCart cart={cart} setCart={setCart} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/scanner"
          element={
            <ProtectedRoute>
              <Header />
              <ScannerPage cart={cart} setCart={setCart} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/checkout"
          element={
            <ProtectedRoute>
              <Header />
              <Checkout cart={cart} setCart={setCart} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/best"
          element={
            <ProtectedRoute>
              <Header />
              <BestSellers setCart={setCart} />
            </ProtectedRoute>
          }
        />

        <Route
          path="/wallet"
          element={
            <ProtectedRoute>
              <Header />
              <Wallet />
            </ProtectedRoute>
          }
        />

        <Route
          path="/success"
          element={
            <ProtectedRoute>
              <Success />
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;

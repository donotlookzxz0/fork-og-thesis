import React from "react";
import { useNavigate } from "react-router-dom";
import { FaShoppingCart } from "react-icons/fa";
import "./landing.css";

const LandingPage = () => {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/login"); // always go to login
  };

  return (
    <div className="landing">
      <div className="landing-card">
        <FaShoppingCart className="landing-icon" />

        <h2 className="landing-title">
          Welcome to GFriend's Korean Mart!
        </h2>

        <p className="landing-description">
          A smart inventory and self-checkout system designed for faster,
          simpler, and more efficient shopping.
        </p>

        <button
          className="landing-btn"
          onClick={handleStart}
        >
          Start Shopping
        </button>
      </div>
    </div>
  );
};

export default LandingPage;

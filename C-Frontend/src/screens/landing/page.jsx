import React from "react";
import { useNavigate } from "react-router-dom";
import "./landing.css";

const LandingPage = () => {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/login"); // always go to login
  };

  return (
    <div className="landing">
      <div className="landing-card">

        {/* Rectangle Logo */}
        <img
          src="/rectangle.png"
          alt="GFriends Korean Mart"
          className="landing-logo"
        />

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

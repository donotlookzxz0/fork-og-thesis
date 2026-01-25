import React from "react";
import { useNavigate } from "react-router-dom";
import "./landing.css";

const LandingPage = () => {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/login");
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

        {/* Intro Text BACK */}
        <h2 className="landing-title">
          Welcome to GFriend's Korean Mart!
        </h2>

        <p className="landing-description">
          Powered by Pi-Mart, an AI-powered general-purpose 
          store system delivering faster checkout, smarter inventory, 
          and ultimate shopping convenience.
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

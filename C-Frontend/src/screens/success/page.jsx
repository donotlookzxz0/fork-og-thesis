import { useEffect } from "react";
import { Link } from "react-router-dom";
import "./success.css";

export default function Success() {
  useEffect(() => {
    // clear cart after successful payment
    localStorage.removeItem("cart");
  }, []);

  return (
    <div className="success-container">
      <div className="success-card">
        <h2>Payment Successful!</h2>
        <p>
          Your payment has been processed. <br /> Thank you for shopping at PiMart.
        </p>

        <Link to="/scanner" className="success-btn">
          Back to Scanner
        </Link>
      </div>
    </div>
  );
}

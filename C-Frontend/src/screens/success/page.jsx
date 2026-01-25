import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import "./success.css";

export default function Success() {
  const { state } = useLocation();
  const totalPrice = state?.totalPrice ?? 0;

  useEffect(() => {
    // clear cart after successful payment
    localStorage.removeItem("cart");
  }, []);

  return (
    <div className="success-container">
      <div className="success-card">
        <h2>Payment Successful!</h2>

        <p>
          Your payment has been processed. <br />
          Thank you for shopping at PiMart.
        </p>

        <h3>
          Total Paid: ₱{totalPrice.toFixed(2)}
        </h3>

        <Link to="/buy" className="success-btn">
          Back to Scanner
        </Link>
      </div>
    </div>
  );
}

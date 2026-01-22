import { useEffect, useState } from "react";
import api from "../api/axios";

function Wallet() {
  const [username, setUsername] = useState("");
  const [balance, setBalance] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchWallet = async () => {
      try {
        const res = await api.get("/payment/wallet/balance");
        setUsername(res.data.username);
        setBalance(res.data.balance);
      } catch (err) {
        console.error(err);
        setError("Failed to load wallet");
      } finally {
        setLoading(false);
      }
    };

    fetchWallet();
  }, []);

  return (
    <div className="container mt-5">
      <div
        className="card p-5 shadow-sm mx-auto"
        style={{ maxWidth: 420, backgroundColor: "#ffffff" }}
      >
        {loading && <p className="text-center">Loading...</p>}

        {error && <p className="text-danger text-center">{error}</p>}

        {!loading && !error && (
          <>
            {/* Greeting */}
            <h4 className="mb-4 text-center">Hi, {username}</h4>

            {/* Balance */}
            <div className="text-center mb-4">
              <p className="text-muted mb-1">Wallet Balance</p>
              <h2 className="fw-bold">
                ₱{Number(balance).toLocaleString("en-PH", {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </h2>
            </div>

            {/* Note */}
            <div
              className="text-center text-muted pt-3"
              style={{
                borderTop: "1px solid #E5E7EB",
                fontSize: 14,
              }}
            >
              To increase your balance, please cash in at G-Friends’
              designated cashier.
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Wallet;

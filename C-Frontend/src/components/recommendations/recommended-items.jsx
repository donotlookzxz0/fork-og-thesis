import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../../api/axios";
import { useCartActions } from "../../hooks/useCartActions";
import "./styles.css";

function RecommendedItems({ setCart }) {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  const [adding, setAdding] = useState(null);
  const [success, setSuccess] = useState(null);

  const { addToCart } = useCartActions(setCart);

  useEffect(() => {
    const fetchRecommendations = async () => {
      setLoading(true);
      try {
        const res = await api.get("/recommendations/me");

        setItems(res.data.recommendations || []);
      } catch (err) {
        console.error("Error loading recommendations:", err);
        setItems([]);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, []);

  const handleAdd = (item) => {
    setAdding(item.barcode);

    setTimeout(() => {
      addToCart(item);
      setAdding(null);
      setSuccess(item.barcode);

      setTimeout(() => setSuccess(null), 1500);
    }, 600); // small UX delay
  };

  return (
    <>
      <div className="go-cart-nav">
        <button className="go-cart-btn" onClick={() => navigate("/cart")}>
          Go to Cart
        </button>
      </div>

      <div className="recommended-container">
        <h1 className="recommended-title">Recommended Items</h1>
        <p className="recommended-subtitle">
          AI-powered product suggestions based on your purchases
        </p>

        {loading ? (
          <p className="loading">Loading recommendations...</p>
        ) : items.length === 0 ? (
          <p className="no-data">Start shopping to get personalized recommendations!.</p>
        ) : (
          <div className="recommended-grid">
            {items.map((item, index) => (
              <div key={index} className="recommended-card">
                <h3 className="item-name">{item.name}</h3>
                <p className="item-price">₱{item.price.toFixed(2)}</p>
                <p className="item-barcode">
                  Barcode: <strong>{item.barcode}</strong>
                </p>

                <button
                  className="add-btn"
                  disabled={adding === item.barcode}
                  onClick={() => handleAdd(item)}
                >
                  {adding === item.barcode
                    ? "Adding..."
                    : success === item.barcode
                      ? "Added"
                      : "Add to Cart"}
                </button>

              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

export default RecommendedItems;
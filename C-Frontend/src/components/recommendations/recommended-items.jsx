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
  const [bestSellers, setBestSellers] = useState([]);
  const [bestLoading, setBestLoading] = useState(true);
  const [bestAdding, setBestAdding] = useState(null);
  const [bestSuccess, setBestSuccess] = useState(null);
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

  useEffect(() => {
    api.get("/ml/item-movement-forecast/best")
      .then((res) => setBestSellers(res.data || []))
      .catch(() => setBestSellers([]))
      .finally(() => setBestLoading(false));
  }, []);

  const handleAdd = (item) => {
    if (item.quantity <= 0) {
      setSuccess(item.barcode);
      setTimeout(() => setSuccess(null), 1500);
      return;
    }
    setAdding(item.barcode);
    setTimeout(() => {
      addToCart(item);
      setAdding(null);
      setSuccess(item.barcode);
      setTimeout(() => setSuccess(null), 1500);
    }, 600);
  };

  const handleBestAdd = (item) => {
    if (item.quantity <= 0) {
      setBestSuccess(item.barcode);
      setTimeout(() => setBestSuccess(null), 1500);
      return;
    }
    setBestAdding(item.barcode);
    setTimeout(() => {
      addToCart(item);
      setBestAdding(null);
      setBestSuccess(item.barcode);
      setTimeout(() => setBestSuccess(null), 1500);
    }, 600);
  };

  return (
    <>
      <div className="go-cart-nav">
        <button className="go-cart-btn" onClick={() => navigate("/cart")}>
          Go to Cart
        </button>
        <a href="#best-sellers" className="best-sellers-link">Best Sellers</a>
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
                  disabled={adding === item.barcode || item.quantity <= 0}
                  onClick={() => handleAdd(item)}
                >
                  {item.quantity <= 0
                    ? "Out of Stock"
                    : adding === item.barcode
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
      <div id="best-sellers" className="recommended-container">
        <h1 className="recommended-title">Best Selling Products</h1>
        <p className="recommended-subtitle">Top products based on overall sales movement</p>
        {bestLoading ? (
          <p className="loading">Loading best sellers...</p>
        ) : bestSellers.length === 0 ? (
          <p className="no-data">No sales data available yet.</p>
        ) : (
          <div className="recommended-grid">
            {bestSellers.map((item, index) => (
              <div key={index} className="recommended-card">
                <h3 className="item-name">{item.name}</h3>
                <p className="item-price">₱{parseFloat(item.price).toFixed(2)}</p>
                <p className="item-barcode">
                  Barcode: <strong>{item.barcode}</strong>
                </p>
                <button
                  className="add-btn"
                  disabled={bestAdding === item.barcode || item.quantity <= 0}
                  onClick={() => handleBestAdd(item)}
                >
                  {item.quantity <= 0
                    ? "Out of Stock"
                    : bestAdding === item.barcode
                      ? "Adding..."
                      : bestSuccess === item.barcode
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
import { useNavigate } from "react-router-dom";
import { CiShoppingCart } from "react-icons/ci";
import { useCartActions } from "../../hooks/useCartActions";
import "./styles.css";

const Items = ({ cart, setCart }) => {
  const navigate = useNavigate();
  const { changeQuantity, deleteItem } = useCartActions(setCart);

  const grandTotal = cart.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  return (
    <div className="cart-container">
      {/* Back button */}
      <div className="cart-nav">
        <button className="back-btn" onClick={() => navigate("/scanner")}>
          ← Back to Scanner
        </button>
      </div>

      <h2 className="cart-title">
        <CiShoppingCart /> Shopping Cart
      </h2>

      {cart.length === 0 ? (
        <p className="cart-empty">No items in your cart yet.</p>
      ) : (
        <>
          {/* Desktop table */}
          <div className="cart-table">
            <table>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Category</th>
                  <th>Price</th>
                  <th>Quantity</th>
                  <th>Total</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {cart.map((item) => (
                  <tr key={item.barcode}>
                    <td>{item.name}</td>
                    <td>{item.category}</td>
                    <td>₱{item.price.toFixed(2)}</td>
                    <td>
                      <div className="qty-controls">
                        <button
                          className="qty-btn minus"
                          onClick={() =>
                            changeQuantity(
                              item.barcode,
                              Math.max(1, item.quantity - 1)
                            )
                          }
                        >
                          -
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={item.quantity}
                          onChange={(e) =>
                            changeQuantity(
                              item.barcode,
                              parseInt(e.target.value) || 1
                            )
                          }
                        />
                        <button
                          className="qty-btn plus"
                          onClick={() =>
                            changeQuantity(item.barcode, item.quantity + 1)
                          }
                        >
                          +
                        </button>
                      </div>
                    </td>
                    <td>₱{(item.price * item.quantity).toFixed(2)}</td>
                    <td>
                      <button
                        className="remove-btn"
                        onClick={() => deleteItem(item.barcode)}
                      >
                        🗑 Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile cards */}
          <div className="cart-cards">
            {cart.map((item) => (
              <div className="cart-card" key={item.barcode}>
                <h4>{item.name}</h4>
                <p>Category: {item.category}</p>
                <p>Price: ₱{item.price.toFixed(2)}</p>

                <div className="qty-controls">
                  <button
                    className="qty-btn minus"
                    onClick={() =>
                      changeQuantity(
                        item.barcode,
                        Math.max(1, item.quantity - 1)
                      )
                    }
                  >
                    -
                  </button>
                  <input
                    type="number"
                    min="1"
                    value={item.quantity}
                    onChange={(e) =>
                      changeQuantity(
                        item.barcode,
                        parseInt(e.target.value) || 1
                      )
                    }
                  />
                  <button
                    className="qty-btn plus"
                    onClick={() =>
                      changeQuantity(item.barcode, item.quantity + 1)
                    }
                  >
                    +
                  </button>
                </div>

                <p className="item-total">
                  Total: ₱{(item.price * item.quantity).toFixed(2)}
                </p>

                <button
                  className="remove-btn"
                  onClick={() => deleteItem(item.barcode)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>

          <h3 className="grand-total">
            Grand Total: ₱{grandTotal.toFixed(2)}
          </h3>

          <button
            className="checkout-btn"
            onClick={() => navigate("/checkout")}
          >
            Proceed to Checkout
          </button>
        </>
      )}
    </div>
  );
};

export default Items;

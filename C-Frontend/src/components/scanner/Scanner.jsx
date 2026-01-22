import { useNavigate } from "react-router-dom";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCamera, faCartShopping, faPaperclip } from "@fortawesome/free-solid-svg-icons";
import { useScanner } from "../../hooks/useScanner";
import { useCartActions } from "../../hooks/useCartActions";
import "./styles.css";

const PageWrapper = ({ children }) => (
  <div className="page-wrapper">{children}</div>
);

const Section = ({ children }) => (
  <div style={{
    marginTop: 24, padding: 24, borderRadius: 16,
    background: "#ffffff", border: "1px solid #E5E7EB",
    boxShadow: "0 6px 16px rgba(0,0,0,.1)"
  }}>{children}</div>
);

const PrimaryButton = ({ children, onClick, style }) => (
  <button
    onClick={onClick}
    style={{
      background: "#ea1e25", color: "#fff", border: "none",
      borderRadius: 8, padding: "12px 18px",
      fontWeight: 500, cursor: "pointer",
      display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
      ...style
    }}
  >
    {children}
  </button>
);

const Scanner = ({ cart, setCart }) => {
  const { addToCart, changeQuantity, deleteItem } = useCartActions(setCart);

  const navigate = useNavigate();
  const {
    videoRef, barcodeInput, setBarcodeInput, nameInput, setNameInput,
    isScanning, setIsScanning, successItem, fetchProduct, suggestions
  } = useScanner({ cart, onAddToCart: addToCart, onQuantityChange: changeQuantity });

  return (
    <PageWrapper>
      <PrimaryButton onClick={() => navigate("/cart")} style={{ marginLeft: "auto" }}>
        <FontAwesomeIcon icon={faCartShopping} /> Go to Cart
      </PrimaryButton>
      
      <h2 style={{ color: "#000", marginTop: 24 }}>
        <FontAwesomeIcon icon={faCamera} /> Barcode Scanner
      </h2>

      <div className="scanner-layout">
        <div className="scanner-column">
          <Section>
            <PrimaryButton
              onClick={() => setIsScanning(s => !s)}
              style={{ background: isScanning ? "#DC2626" : "#000", marginBottom: 16 }}
            >
              {isScanning ? "Stop Camera" : "Start Camera"}
            </PrimaryButton>

            <div className="scanner-actions">
              <input className="scanner-input" value={barcodeInput}
                onChange={e => setBarcodeInput(e.target.value)}
                placeholder="Enter barcode manually" />
              <PrimaryButton onClick={() => fetchProduct(barcodeInput)}>Add</PrimaryButton>
            </div>

            {isScanning && (
              <div className="scanner-video-wrapper">
                <video ref={videoRef} className="scanner-video" />
                <div className="barcode-overlay"><div className="barcode-box" /></div>
              </div>
            )}

            <input className="scanner-input" style={{ marginTop: 16 }}
              value={nameInput} onChange={e => setNameInput(e.target.value)}
              placeholder="Add item by name" />

            {nameInput && (
              <div className="suggestions">
                {suggestions.map(item => (
                  <div key={item.barcode} className="suggestion-item"
                    onClick={() => { fetchProduct(item.barcode); setNameInput(""); }}>
                    {item.name}
                  </div>
                ))}
              </div>
            )}
          </Section>
        </div>

        <div className="scanner-column">
          <Section>
            <h3 style={{ color: "#000" }}>
              <FontAwesomeIcon icon={faPaperclip} /> Scanned Items
            </h3>

            {!cart.length ? <p style={{ color: "#6B7280" }}>No items yet...</p> :
              cart.map(item => (
                <div key={item.barcode} style={{ borderBottom: "1px solid #E5E7EB", padding: "12px 0" }}>
                  <strong>{item.name}</strong>
                  <p>₱{item.price.toFixed(2)}</p>
                  <p style={{ color: "#6B7280" }}>{item.category}</p>

                  <PrimaryButton
                    onClick={() => deleteItem(item.barcode)}
                    style={{ background: "#ea1e25" }}
                  >
                    Remove
                  </PrimaryButton>
                </div>
              ))
            }
          </Section>
        </div>
      </div>

      {successItem && (
        <div className="success-modal">
          <div className="success-box">
            <h3>Item Added</h3>
            <strong>{successItem.name}</strong>
          </div>
        </div>
      )}
    </PageWrapper>
  );
};

export default Scanner;

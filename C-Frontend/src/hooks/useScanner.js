import { useState, useRef, useEffect, useCallback } from "react";
import { BrowserMultiFormatReader } from "@zxing/browser";
import api from "../api/axios";

export const useScanner = ({ cart, onAddToCart, onQuantityChange }) => {
  const videoRef = useRef(null);
  const readerRef = useRef(null);
  const controlsRef = useRef(null);
  const lastScannedRef = useRef(null);

  const [barcodeInput, setBarcodeInput] = useState("");
  const [nameInput, setNameInput] = useState("");
  const [items, setItems] = useState([]);          // 🔑 all items cache
  const [isScanning, setIsScanning] = useState(false);
  const [successItem, setSuccessItem] = useState(null);

  /* =======================
     LOAD ALL ITEMS ONCE
     (FOR NAME SEARCH)
  ======================= */
  useEffect(() => {
    const loadItems = async () => {
      try {
        const res = await api.get("/items");
        console.log("SCANNER ITEMS:", res.data);   // 🔥 debug — remove later
        setItems(res.data);
      } catch (e) {
        console.error("Failed to load items for scanner", e);
      }
    };

    loadItems();
  }, []);

  /* =======================
     FETCH PRODUCT BY BARCODE
     (CAMERA + MANUAL INPUT)
  ======================= */
  const fetchProduct = useCallback(
    async (barcode, { resumeScan = false } = {}) => {
      if (!barcode) return;

      try {
        const { data } = await api.get(`/items/barcode/${barcode}`);
        const product = { ...data, price: parseFloat(data.price) };

        const existing = cart.find(i => i.barcode === product.barcode);

        if (existing) {
          onQuantityChange(product.barcode, existing.quantity + 1);
        } else {
          onAddToCart({ ...product, quantity: 1 });
        }

        // UI feedback
        setSuccessItem(product);
        setIsScanning(false);

        // Clear inputs
        setBarcodeInput("");
        setNameInput("");

        setTimeout(() => {
          setSuccessItem(null);
          if (resumeScan) setIsScanning(true);
        }, 1200);

      } catch (e) {
        console.error("Fetch product failed", e);
        alert("Item not found");
      }
    },
    [cart, onAddToCart, onQuantityChange]
  );

  /* =======================
     CAMERA SCANNER
  ======================= */
  useEffect(() => {
    if (!isScanning) {
      controlsRef.current?.stop();
      controlsRef.current = null;
      return;
    }

    const reader = new BrowserMultiFormatReader();
    readerRef.current = reader;

    reader
      .decodeFromVideoDevice(null, videoRef.current, result => {
        if (!result) return;

        const code = result.getText();

        // Prevent duplicate scans
        if (code !== lastScannedRef.current) {
          lastScannedRef.current = code;
          fetchProduct(code, { resumeScan: true });
        }
      })
      .then(controls => {
        controlsRef.current = controls;
      })
      .catch(err => {
        console.error("Camera error:", err);
      });

    return () => {
      controlsRef.current?.stop();
      controlsRef.current = null;
    };
  }, [isScanning, fetchProduct]);

  /* =======================
     🔥 SMART NAME SUGGESTIONS
     (LOCAL FILTER, NO API)
  ======================= */
  const suggestions = (() => {
    const q = nameInput.trim().toLowerCase();
    if (!q) return [];

    return items
      .map(i => {
        const name = i.name.toLowerCase();
        let score = 0;

        // Best match: starts with
        if (name.startsWith(q)) score = 3;
        // Medium: word boundary match
        else if (name.includes(" " + q)) score = 2;
        // Weak: anywhere match
        else if (name.includes(q)) score = 1;

        return { ...i, _score: score };
      })
      .filter(i => i._score > 0)
      .sort((a, b) => b._score - a._score)
      .slice(0, 5);   // limit to 5 suggestions
  })();

  return {
    videoRef,
    barcodeInput,
    setBarcodeInput,
    nameInput,
    setNameInput,
    isScanning,
    setIsScanning,
    successItem,
    fetchProduct,
    suggestions,
  };
};

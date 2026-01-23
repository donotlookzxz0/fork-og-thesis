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
  const [items, setItems] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [successItem, setSuccessItem] = useState(null);

  /* =======================
     LOAD ITEMS FOR SEARCH
  ======================= */
  useEffect(() => {
    const loadItems = async () => {
      try {
        const res = await api.get("/items");
        console.log("SCANNER ITEMS:", res.data);   // 🔥 IMPORTANT DEBUG
        setItems(res.data);
      } catch (e) {
        console.error("Failed to load items for scanner", e);
      }
    };

    loadItems();
  }, []);

  /* =======================
     FETCH PRODUCT BY BARCODE
  ======================= */
  const fetchProduct = useCallback(
    async (barcode, { resumeScan = false } = {}) => {
      if (!barcode) return;
      try {
        const { data } = await api.get(`/items/barcode/${barcode}`);
        const product = { ...data, price: parseFloat(data.price) };

        const existing = cart.find(i => i.barcode === product.barcode);
        existing
          ? onQuantityChange(product.barcode, existing.quantity + 1)
          : onAddToCart({ ...product, quantity: 1 });

        setSuccessItem(product);
        setIsScanning(false);

        setTimeout(() => {
          setSuccessItem(null);
          if (resumeScan) setIsScanning(true);
        }, 1500);
      } catch (e) {
        console.error("Fetch product failed", e);
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
     SMART NAME SUGGESTIONS
  ======================= */
  const suggestions = items
    .filter(i =>
      i.name.toLowerCase().includes(nameInput.toLowerCase())
    )
    .slice(0, 5);

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

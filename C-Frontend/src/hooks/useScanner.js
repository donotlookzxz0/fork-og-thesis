import { useState, useRef, useEffect, useCallback } from "react";
import { BrowserMultiFormatReader } from "@zxing/browser";
import api from "../api/axios";

export const useScanner = ({ cart, onAddToCart, onQuantityChange }) => {
  const videoRef = useRef(null);
  const readerRef = useRef(null);
  const controlsRef = useRef(null);
  const lastScannedRef = useRef(null);;

  const [barcodeInput, setBarcodeInput] = useState("");
  const [nameInput, setNameInput] = useState("");
  const [items, setItems] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [successItem, setSuccessItem] = useState(null);

  // Fetch all items once
  useEffect(() => {
    api.get("/items").then(res => setItems(res.data));
  }, []);

  const fetchProduct = useCallback(
    async (barcode, { resumeScan = false } = {}) => {
      if (!barcode) return;
      try {
        const { data } = await api.get(`items/barcode/${barcode}`);
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
        console.error(e);
      }
    },
    [cart, onAddToCart, onQuantityChange]
  );

  // Scanner initialization
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

  // Name suggestions
  const suggestions = items.filter(i =>
    i.name.toLowerCase().startsWith(nameInput.toLowerCase())
  );

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

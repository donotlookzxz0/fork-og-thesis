import { useState, useRef, useEffect, useCallback, useMemo } from "react";
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

  /* LOAD ITEMS ONCE */
  useEffect(() => {
    api
      .get("/items")
      .then((res) => setItems(res.data || []))
      .catch(() => setItems([]));
  }, []);

  /* FETCH PRODUCT */
  const fetchProduct = useCallback(
    async (barcode, { resumeScan = false } = {}) => {
      if (!barcode) return;

      try {
        const { data } = await api.get(`/items/barcode/${barcode}`);
        const product = { ...data, price: parseFloat(data.price) };

        const existing = cart.find((i) => i.barcode === product.barcode);
        existing
          ? onQuantityChange(product.barcode, existing.quantity + 1)
          : onAddToCart({ ...product, quantity: 1 });

        setSuccessItem(product);
        setIsScanning(false);
        setBarcodeInput("");
        setNameInput("");

        setTimeout(() => {
          setSuccessItem(null);
          if (resumeScan) setIsScanning(true);
        }, 1200);
      } catch {
        alert("Item not found");
      }
    },
    [cart, onAddToCart, onQuantityChange]
  );

  /* CAMERA */
  useEffect(() => {
    if (!isScanning) {
      controlsRef.current?.stop();
      return;
    }

    const reader = new BrowserMultiFormatReader();
    readerRef.current = reader;

    reader.decodeFromVideoDevice(null, videoRef.current, (result) => {
      if (!result) return;
      const code = result.getText();
      if (code !== lastScannedRef.current) {
        lastScannedRef.current = code;
        fetchProduct(code, { resumeScan: true });
      }
    });

    return () => controlsRef.current?.stop();
  }, [isScanning, fetchProduct]);

  /* SMART NAME SEARCH */
  const suggestions = useMemo(() => {
    if (!nameInput) return [];
    const q = nameInput.toLowerCase().trim();

    return items
      .map((i) => {
        const name = i.name.toLowerCase();
        let score = 0;
        if (name.startsWith(q)) score = 3;
        else if (name.includes(" " + q)) score = 2;
        else if (name.includes(q)) score = 1;
        return { ...i, _score: score };
      })
      .filter((i) => i._score > 0)
      .sort((a, b) => b._score - a._score)
      .slice(0, 6);
  }, [nameInput, items]);

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

export const useCartActions = (setCart) => {
  const addToCart = (product) => {
    setCart((prev) =>
      prev.find((i) => i.barcode === product.barcode)
        ? prev.map((i) =>
            i.barcode === product.barcode
              ? { ...i, quantity: i.quantity + 1 }
              : i
          )
        : [...prev, { ...product, quantity: 1 }]
    );
  };

  const changeQuantity = (barcode, qty) => {
    setCart((prev) =>
      prev.map((i) =>
        i.barcode === barcode
          ? { ...i, quantity: Math.max(1, qty) }
          : i
      )
    );
  };

  const deleteItem = (barcode) => {
    setCart((prev) => prev.filter((i) => i.barcode !== barcode));
  };

  return {
    addToCart,
    changeQuantity,
    deleteItem,
  };
};

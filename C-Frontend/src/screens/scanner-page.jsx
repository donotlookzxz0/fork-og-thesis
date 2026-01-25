import Scanner from "../components/scanner/item-scanner";

const ScannerPage = ({ cart, setCart }) => {
  return <Scanner cart={cart} setCart={setCart} />;
};

export default ScannerPage;

import Items from "../components/cart/items";

const ViewCart = ({ cart, setCart }) => {
  return <Items cart={cart} setCart={setCart} />;
};

export default ViewCart;

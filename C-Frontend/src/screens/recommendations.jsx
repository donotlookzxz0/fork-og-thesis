import RecommendedItems from "../components/recommendations/recommended-items"

function Recommendations({cart, setCart}) {
  return (
    <div>
        <RecommendedItems cart={cart} setCart={setCart}/>
    </div>
  )
}

export default Recommendations
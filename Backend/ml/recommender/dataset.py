from collections import defaultdict
from models.sales_transaction import SalesTransaction
from sqlalchemy.orm import joinedload

def build_interactions():
    user_item = defaultdict(lambda: defaultdict(int))

    transactions = (
        SalesTransaction.query
        .options(joinedload(SalesTransaction.items))
        .all()
    )
    
    for tx in transactions:
        for ti in tx.items:
            user_item[tx.user_id][ti.item_id] += ti.quantity

    return user_item

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date, timedelta

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from models.ai_stockout_risk import AIStockoutRisk


LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class StockoutDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


def run_stockout_risk_forecast():
    today = date.today()
    cutoff_date = today - timedelta(days=30)

    rows = (
        db.session.query(
            Item.id.label("item_id"),
            Item.name.label("item_name"),
            Item.category.label("category"),
            Item.quantity.label("current_stock"),
            SalesTransactionItem.quantity.label("sold_qty"),
            SalesTransaction.date.label("date"),
        )
        .select_from(Item)
        .outerjoin(SalesTransactionItem, Item.id == SalesTransactionItem.item_id)
        .outerjoin(SalesTransaction, SalesTransaction.id == SalesTransactionItem.transaction_id)
        .filter(
            (SalesTransaction.date >= cutoff_date) | (SalesTransaction.date == None)
        )
        .all()
    )

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=[
        "item_id",
        "item_name",
        "category",
        "current_stock",
        "sold_qty",
        "date"
    ])

    features = []
    labels = []
    meta = []

    for item_id, g in df.groupby("item_id"):

        total_sold_30d = float(g["sold_qty"].fillna(0).sum())
        current_stock = int(g["current_stock"].iloc[0])

        if total_sold_30d == 0:
            features.append([current_stock, 0.0, 100.0])
            labels.append(LABEL_MAP["Low"])

            meta.append({
                "item_id": int(item_id),
                "item_name": g["item_name"].iloc[0],
                "category": g["category"].iloc[0],
                "current_stock": current_stock,
                "avg_daily_sales": 0,
                "days_of_stock_left": 100
            })
            continue

        initial_stock = total_sold_30d + current_stock
        if initial_stock <= 0:
            continue

        sold_percentage = (total_sold_30d / initial_stock) * 100
        remaining_percentage = 100 - sold_percentage

        if sold_percentage <= 33:
            label = "Low"
        elif sold_percentage <= 66:
            label = "Medium"
        else:
            label = "High"

        features.append([current_stock, sold_percentage, remaining_percentage])
        labels.append(LABEL_MAP[label])

        meta.append({
            "item_id": int(item_id),
            "item_name": g["item_name"].iloc[0],
            "category": g["category"].iloc[0],
            "current_stock": current_stock,
            "avg_daily_sales": total_sold_30d / 30,
            "days_of_stock_left": remaining_percentage
        })

    if len(features) < 2:
        return None

    dataset = StockoutDataset(features, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = StockoutNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []

    with torch.no_grad():
        for x in features:
            logits = model(torch.tensor(x, dtype=torch.float32))
            preds.append(torch.argmax(logits).item())

    true = torch.tensor(labels)
    pred = torch.tensor(preds)

    accuracy = (true == pred).float().mean().item()
    risk_mae = torch.abs(true - pred).float().mean().item()

    def precision_recall_f1(t, p, cls):
        tp = ((t == cls) & (p == cls)).sum().item()
        fp = ((t != cls) & (p == cls)).sum().item()
        fn = ((t == cls) & (p != cls)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    precision = {}
    recall = {}
    f1_scores = []

    for name, cls in LABEL_MAP.items():
        pr, rc, f1 = precision_recall_f1(true, pred, cls)
        precision[name] = float(pr)
        recall[name] = float(rc)
        f1_scores.append(f1)

    macro_f1 = float(sum(f1_scores) / len(f1_scores))

    global_metrics = {
        "accuracy": float(accuracy),
        "risk_mae": float(risk_mae),
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "total_items": len(labels)
    }

    category_metrics = {}

    for i, info in enumerate(meta):
        cat = info["category"]

        if cat not in category_metrics:
            category_metrics[cat] = {
                "true": [],
                "pred": []
            }

        category_metrics[cat]["true"].append(true[i].item())
        category_metrics[cat]["pred"].append(pred[i].item())

    for cat in category_metrics:
        t = torch.tensor(category_metrics[cat]["true"])
        p = torch.tensor(category_metrics[cat]["pred"])

        acc = (t == p).float().mean().item()
        mae = torch.abs(t - p).float().mean().item()

        prec = {}
        rec = {}
        f1s = []

        for name, cls in LABEL_MAP.items():
            pr, rc, f1 = precision_recall_f1(t, p, cls)
            prec[name] = float(pr)
            rec[name] = float(rc)
            f1s.append(f1)

        category_metrics[cat] = {
            "accuracy": float(acc),
            "risk_mae": float(mae),
            "macro_f1": float(sum(f1s) / len(f1s)),
            "precision": prec,
            "recall": rec,
            "total_items": int(len(t))
        }

    AIStockoutRisk.query.delete()

    with torch.no_grad():
        for x, info in zip(features, meta):
            logits = model(torch.tensor(x, dtype=torch.float32))
            pred_cls = torch.argmax(logits).item()
            risk = INV_LABEL_MAP[pred_cls]

            db.session.add(AIStockoutRisk(
                item_id=info["item_id"],
                item_name=info["item_name"],
                category=info["category"],
                current_stock=info["current_stock"],
                avg_daily_sales=info["avg_daily_sales"],
                days_of_stock_left=info["days_of_stock_left"],
                risk_level=risk
            ))

    db.session.commit()

    return {
        "success": True,
        "metrics": category_metrics,
        "global_metrics": global_metrics
    }
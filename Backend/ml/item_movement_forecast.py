import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date, timedelta

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item
from models.ai_item_movement import AIItemMovement


LABEL_MAP = {"Slow": 0, "Medium": 1, "Fast": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class MovementDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MovementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)


def run_item_movement_forecast():
    today = date.today()
    cutoff_date = today - timedelta(days=30)

    rows = (
        db.session.query(
            Item.id.label("item_id"),
            Item.name.label("item_name"),
            Item.category.label("category"),
            SalesTransaction.date.label("date"),
            SalesTransactionItem.quantity.label("quantity"),
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
        "item_id", "item_name", "category", "date", "quantity"
    ])

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    features = []
    labels = []
    meta = []

    for item_id, g in df.groupby("item_id"):
        total_sold = int(g["quantity"].fillna(0).sum())
        avg_daily_sales = float(total_sold / 30)

        if g["date"].notna().sum() == 0:
            features.append([0.0, 999])
            labels.append(LABEL_MAP["Slow"])

            meta.append({
                "item_id": int(item_id),
                "item_name": g["item_name"].iloc[0],
                "category": g["category"].iloc[0],
                "avg_daily_sales": 0,
                "days_since_last_sale": 999
            })
            continue

        days_since_last_sale = int((today - g["date"].max()).days)

        if (
            avg_daily_sales >= 0.5
            and total_sold >= 15
            and days_since_last_sale <= 3
        ):
            label = "Fast"
        elif (
            avg_daily_sales >= 0.2
            and total_sold >= 5
            and days_since_last_sale <= 7
        ):
            label = "Medium"
        else:
            label = "Slow"

        features.append([
            avg_daily_sales,
            days_since_last_sale
        ])
        labels.append(LABEL_MAP[label])

        meta.append({
            "item_id": int(item_id),
            "item_name": g["item_name"].iloc[0],
            "category": g["category"].iloc[0],
            "avg_daily_sales": avg_daily_sales,
            "days_since_last_sale": days_since_last_sale
        })

    if len(features) < 2:
        return None

    dataset = MovementDataset(features, labels)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = MovementNet()
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
            pred = torch.argmax(logits).item()
            preds.append(pred)

    true = torch.tensor(labels)
    pred = torch.tensor(preds)

    total_items = len(labels)
    accuracy = (true == pred).float().mean().item()
    movement_mae = torch.abs(true - pred).float().mean().item()

    def compute_class_metrics(true_labels, pred_labels):
        precision = {}
        recall = {}
        f1_scores = []

        for name, idx in LABEL_MAP.items():
            tp = ((pred_labels == idx) & (true_labels == idx)).sum().item()
            fp = ((pred_labels == idx) & (true_labels != idx)).sum().item()
            fn = ((pred_labels != idx) & (true_labels == idx)).sum().item()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

            precision[name] = float(p)
            recall[name] = float(r)
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return precision, recall, float(macro_f1)

    precision_global, recall_global, macro_f1_global = compute_class_metrics(true, pred)

    category_metrics = {}
    meta_df = pd.DataFrame(meta)
    meta_df["true"] = true.numpy()
    meta_df["pred"] = pred.numpy()

    for category, g in meta_df.groupby("category"):
        t = torch.tensor(g["true"].values)
        p = torch.tensor(g["pred"].values)

        acc = (t == p).float().mean().item()
        mae = torch.abs(t - p).float().mean().item()
        prec, rec, mf1 = compute_class_metrics(t, p)

        category_metrics[category] = {
            "accuracy": float(acc),
            "macro_f1": float(mf1),
            "movement_mae": float(mae),
            "precision": prec,
            "recall": rec,
            "total_items": int(len(g))
        }

    global_metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1_global),
        "movement_mae": float(movement_mae),
        "precision": precision_global,
        "recall": recall_global,
        "total_items": int(total_items)
    }

    AIItemMovement.query.delete()

    with torch.no_grad():
        for x, info in zip(features, meta):
            logits = model(torch.tensor(x, dtype=torch.float32))
            pred = torch.argmax(logits).item()
            label = INV_LABEL_MAP[pred]

            db.session.add(AIItemMovement(
                item_id=info["item_id"],
                item_name=info["item_name"],
                category=info["category"],
                avg_daily_sales=info["avg_daily_sales"],
                days_since_last_sale=info["days_since_last_sale"],
                movement_class=label
            ))

    db.session.commit()

    return {
        "success": True,
        "metrics": category_metrics,
        "global_metrics": global_metrics
    }
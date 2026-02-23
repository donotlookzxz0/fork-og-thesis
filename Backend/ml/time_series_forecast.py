import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TSDataset(Dataset):
    def __init__(self, data, seq_len):
        self.X, self.y = [], []

        for i in range(len(data) - seq_len):
            self.X.append(data[i:i + seq_len])
            self.y.append(data[i + seq_len])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class LSTM(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.lstm = nn.LSTM(features, 16, batch_first=True)
        self.fc = nn.Linear(16, features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


def run_time_series_forecast():
    print("\n[ML] Stable category-wise demand prediction started")

    rows = (
        db.session.query(
            SalesTransaction.date.label("date"),
            Item.category.label("category"),
            SalesTransactionItem.quantity.label("quantity"),
        )
        .select_from(SalesTransaction)
        .join(
            SalesTransactionItem,
            SalesTransaction.id == SalesTransactionItem.transaction_id
        )
        .join(
            Item,
            Item.id == SalesTransactionItem.item_id
        )
        .all()
    )

    if not rows:
        print("[ML] No sales data found")
        return None

    df = pd.DataFrame(rows, columns=["date", "category", "quantity"])
    df["date"] = pd.to_datetime(df["date"]).dt.date

    daily = (
        df.groupby(["date", "category"])["quantity"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if len(daily) < 30:
        print("[ML] Not enough data")
        return None

    results = {
        "tomorrow": {},
        "next_7_days": {},
        "next_30_days": {}
    }

    metrics = {
        "tomorrow": {},
        "next_7_days": {},
        "next_30_days": {}
    }

    for category in daily.columns:
        print(f"[ML] Training category: {category}")

        series = daily[[category]].values
        values = np.log1p(series)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        SEQ_LEN = min(14, len(scaled) - 31)

        if SEQ_LEN <= 2:
            continue

        ds = TSDataset(scaled, SEQ_LEN)
        loader = DataLoader(ds, batch_size=8, shuffle=False)

        model = LSTM(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        for epoch in range(40):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()

        preds_1, actual_1 = [], []
        preds_7, actual_7 = [], []
        preds_30, actual_30 = [], []

        for i in range(len(scaled) - SEQ_LEN - 30):
            seq = torch.tensor(
                scaled[i:i+SEQ_LEN], dtype=torch.float32
            ).unsqueeze(0)

            future = []
            with torch.no_grad():
                temp = seq.clone()
                for _ in range(30):
                    out = model(temp)
                    future.append(out.numpy()[0])
                    temp = torch.cat(
                        [temp[:, 1:, :], out.unsqueeze(1)],
                        dim=1
                    )

            future = scaler.inverse_transform(np.array(future))
            future = np.expm1(future)

            real = scaler.inverse_transform(
                scaled[i+SEQ_LEN:i+SEQ_LEN+30]
            )
            real = np.expm1(real)

            preds_1.append(future[0][0])
            actual_1.append(real[0][0])

            preds_7.append(np.sum(future[:7]))
            actual_7.append(np.sum(real[:7]))

            preds_30.append(np.sum(future[:30]))
            actual_30.append(np.sum(real[:30]))

        def calc_metrics(a, p):
            a = np.array(a)
            p = np.array(p)
            mae = mean_absolute_error(a, p)
            rmse = np.sqrt(mean_squared_error(a, p))
            mask = a != 0
            mape = (
                np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100
                if np.any(mask) else 0.0
            )
            return {
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape)
            }

        metrics["tomorrow"][category] = calc_metrics(actual_1, preds_1)
        metrics["next_7_days"][category] = calc_metrics(actual_7, preds_7)
        metrics["next_30_days"][category] = calc_metrics(actual_30, preds_30)

        def predict(days):
            seq = torch.tensor(
                scaled[-SEQ_LEN:], dtype=torch.float32
            ).unsqueeze(0)

            preds = []

            for _ in range(days):
                with torch.no_grad():
                    nxt = model(seq)

                preds.append(nxt.numpy()[0])
                seq = torch.cat(
                    [seq[:, 1:, :], nxt.unsqueeze(1)],
                    dim=1
                )

            preds = scaler.inverse_transform(np.array(preds))
            preds = np.expm1(preds)
            return np.clip(preds, 0, None)

        results["tomorrow"][category] = int(round(predict(1)[0][0]))
        results["next_7_days"][category] = int(np.sum(predict(7)))
        results["next_30_days"][category] = int(np.sum(predict(30)))

    return {
        "results": results,
        "metrics": metrics
    }
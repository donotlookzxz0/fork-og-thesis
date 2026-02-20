# ml/time_series_forecast.py
# STABLE + CATEGORY-WISE TIME-SERIES DEMAND FORECAST
# Improvements:
# - Train per category (no averaging issue)
# - Smaller LSTM for small POS datasets
# - Uses newest data for prediction
# - Stable deterministic training

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from db import db
from models.sales_transaction import SalesTransaction
from models.sales_transaction_item import SalesTransactionItem
from models.item import Item


# ===============================
# 0️⃣ FIX RANDOMNESS
# ===============================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===============================
# DATASET CLASS
# ===============================
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


# ===============================
# MODEL
# ===============================
class LSTM(nn.Module):
    def __init__(self, features):
        super().__init__()
        # 🔥 smaller network = more stable with small data
        self.lstm = nn.LSTM(features, 16, batch_first=True)
        self.fc = nn.Linear(16, features)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ===============================
# MAIN FORECAST FUNCTION
# ===============================
def run_time_series_forecast():
    print("\n[ML] Stable category-wise demand prediction started")

    # ===============================
    # 1️⃣ LOAD DATA
    # ===============================
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

    # ===============================
    # 2️⃣ DAILY CATEGORY SERIES
    # ===============================
    daily = (
        df.groupby(["date", "category"])["quantity"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if len(daily) < 14:
        print("[ML] Not enough data")
        return None

    results = {
        "tomorrow": {},
        "next_7_days": {},
        "next_30_days": {}
    }

    # ===============================
    # 3️⃣ TRAIN PER CATEGORY
    # ===============================
    for category in daily.columns:

        print(f"[ML] Training category: {category}")

        series = daily[[category]].values  # shape (days,1)

        # log transform
        values = np.log1p(series)

        # train split
        split = int(len(values) * 0.8)
        train_data = values[:split]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)

        # 🔥 shorter window works better for POS data
        SEQ_LEN = min(14, len(train_scaled) - 1)

        if SEQ_LEN <= 2:
            print(f"[ML] Skipping {category} (not enough sequence data)")
            continue

        train_ds = TSDataset(train_scaled, SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)

        # ===============================
        # MODEL INIT
        # ===============================
        model = LSTM(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        # ===============================
        # TRAIN
        # ===============================
        for epoch in range(40):
            total_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # ===============================
        # 4️⃣ PREDICT USING FULL DATA
        # ===============================
        full_scaled = scaler.transform(values)

        def predict(days):
            model.eval()

            seq = torch.tensor(
                full_scaled[-SEQ_LEN:], dtype=torch.float32
            ).unsqueeze(0)

            preds = []

            for _ in range(days):
                with torch.no_grad():
                    next_step = model(seq)

                preds.append(next_step.numpy()[0])

                seq = torch.cat(
                    [seq[:, 1:, :], next_step.unsqueeze(1)],
                    dim=1
                )

            preds = scaler.inverse_transform(np.array(preds))
            preds = np.expm1(preds)
            return np.clip(preds, 0, None)

        pred_1 = predict(1)[0][0]
        pred_7 = predict(7)[:, 0]
        pred_30 = predict(30)[:, 0]

        results["tomorrow"][category] = int(round(pred_1))
        results["next_7_days"][category] = int(pred_7.sum())
        results["next_30_days"][category] = int(pred_30.sum())

    return results
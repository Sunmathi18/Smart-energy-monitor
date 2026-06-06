# train_model.py
# This file trains an AI to detect energy anomalies

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ── Load Data ──────────────────────────────────────────
print("📂 Loading energy data...")
df = pd.read_csv("data/energy_data.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Feature Engineering ────────────────────────────────
# Give the AI extra info to learn from
df["hour"]      = df["timestamp"].dt.hour
df["day"]       = df["timestamp"].dt.day
df["dayofweek"] = df["timestamp"].dt.dayofweek

# ── MODEL 1: Anomaly Detection (Isolation Forest) ──────
print("🤖 Training Anomaly Detection AI...")

features = ["fan", "ac", "light", "fridge", "washing_machine", "total_wh"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

anomaly_model = IsolationForest(
    contamination=0.03,  # expects ~3% anomalies
    random_state=42
)
anomaly_model.fit(X_scaled)

# Add predictions to dataframe (-1 = anomaly, 1 = normal)
df["anomaly"] = anomaly_model.predict(X_scaled)
df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "⚠️ ANOMALY"})

anomaly_count = (df["anomaly"] == -1).sum()
print(f"✅ Anomaly model trained! Found {anomaly_count} anomalies in 720 hours")

# ── MODEL 2: Energy Prediction (Linear Regression) ─────
print("📈 Training Energy Prediction AI...")

pred_features = ["hour", "day", "dayofweek", "fan", "ac", "light", "fridge"]
X_pred = df[pred_features]
y_pred = df["total_wh"]

pred_model = LinearRegression()
pred_model.fit(X_pred, y_pred)
print("✅ Prediction model trained!")

# ── Save Models ────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/anomaly_model.pkl", "wb") as f:
    pickle.dump(anomaly_model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/pred_model.pkl", "wb") as f:
    pickle.dump(pred_model, f)

# Save updated data with anomaly labels
df.to_csv("data/energy_data_labeled.csv", index=False)

print("\n🎯 All models saved to models/ folder!")
print(f"\n📊 Anomaly Summary:")
print(df["anomaly_label"].value_counts())
# dashboard.py
# Creates visual dashboard of energy data + AI results

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
import numpy as np

# ── Load Data ──────────────────────────────────────────
print("📂 Loading labeled data...")
df = pd.read_csv("data/energy_data_labeled.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── Load Prediction Model ──────────────────────────────
with open("models/pred_model.pkl", "rb") as f:
    pred_model = pickle.load(f)

# ── Predict next 24 hours ──────────────────────────────
last_row = df.iloc[-1]
future_hours = pd.DataFrame({
    "hour":      list(range(24)),
    "day":       [last_row["day"]] * 24,
    "dayofweek": [last_row["dayofweek"]] * 24,
    "fan":       [df["fan"].mean()] * 24,
    "ac":        [df["ac"].mean()] * 24,
    "light":     [df["light"].mean()] * 24,
    "fridge":    [df["fridge"].mean()] * 24,
})
future_hours["predicted_wh"] = pred_model.predict(future_hours)

# ── Separate normal vs anomaly ─────────────────────────
normal    = df[df["anomaly_label"] == "Normal"]
anomalies = df[df["anomaly_label"] == "⚠️ ANOMALY"]

# ── Build Dashboard ────────────────────────────────────
sns.set_theme(style="darkgrid")
fig = plt.figure(figsize=(18, 12))
fig.suptitle("🏠 AI Smart Home Energy Monitor Dashboard",
             fontsize=20, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Graph 1: Total energy over time + anomalies ────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df["timestamp"], df["total_wh"],
         color="steelblue", linewidth=0.8, label="Energy Usage (Wh)")
ax1.scatter(anomalies["timestamp"], anomalies["total_wh"],
            color="red", zorder=5, s=40, label="⚠️ Anomaly Detected")
ax1.set_title("Total Energy Usage with Anomaly Detection", fontsize=13)
ax1.set_xlabel("Date")
ax1.set_ylabel("Watt-Hours")
ax1.legend()

# ── Graph 2: Device-wise average usage ─────────────────
ax2 = fig.add_subplot(gs[1, 0])
device_avg = df[["fan","ac","light","fridge","washing_machine"]].mean()
colors = ["#4CAF50","#2196F3","#FFC107","#9C27B0","#FF5722"]
ax2.bar(device_avg.index, device_avg.values, color=colors, edgecolor="white")
ax2.set_title("Average Power Usage per Device", fontsize=13)
ax2.set_xlabel("Device")
ax2.set_ylabel("Average Watts")
ax2.tick_params(axis="x", rotation=20)

# ── Graph 3: Next 24-hour energy prediction ────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(future_hours["hour"], future_hours["predicted_wh"],
         color="orange", linewidth=2, marker="o", markersize=4)
ax3.fill_between(future_hours["hour"], future_hours["predicted_wh"],
                 alpha=0.2, color="orange")
ax3.set_title("🔮 AI Predicted Energy — Next 24 Hours", fontsize=13)
ax3.set_xlabel("Hour of Day")
ax3.set_ylabel("Predicted Watt-Hours")

# ── Save + Show ────────────────────────────────────────
plt.savefig("data/dashboard.png", dpi=150, bbox_inches="tight")
print("✅ Dashboard saved to data/dashboard.png")
plt.show()
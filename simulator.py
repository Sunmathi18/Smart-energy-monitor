# simulator.py
# This file generates fake smart home energy data

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ── Settings ──────────────────────────────────────────
DAYS = 30          # how many days of data to simulate
INTERVAL = 1       # every 1 hour

# ── Device average power usage in Watts ───────────────
DEVICES = {
    "fan":             {"avg": 75,   "std": 10},
    "ac":              {"avg": 1500, "std": 200},
    "light":           {"avg": 60,   "std": 5},
    "fridge":          {"avg": 150,  "std": 20},
    "washing_machine": {"avg": 500,  "std": 50},
}

def simulate_usage(days=DAYS):
    records = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for hour in range(days * 24):
        timestamp = start_time + timedelta(hours=hour)
        hour_of_day = timestamp.hour
        row = {"timestamp": timestamp}

        for device, specs in DEVICES.items():
            # Devices are OFF during certain hours (simulate real life)
            if device == "ac" and hour_of_day not in range(10, 23):
                row[device] = 0.0  # AC off at night/early morning
            elif device == "washing_machine" and hour_of_day not in range(8, 12):
                row[device] = 0.0  # Washing machine only morning
            elif device == "light" and hour_of_day in range(6, 18):
                row[device] = 0.0  # Lights off during daytime
            else:
                # Generate random realistic usage
                usage = np.random.normal(specs["avg"], specs["std"])
                row[device] = round(max(0, usage), 2)

        # Add some ANOMALIES (abnormal spikes) — AI will detect these later
        if np.random.rand() < 0.03:  # 3% chance of anomaly per hour
            spike_device = np.random.choice(list(DEVICES.keys()))
            row[spike_device] *= 3   # Triple the usage — clear anomaly!

        # Total energy for that hour (in Watt-hours)
        row["total_wh"] = round(
            sum(row[d] for d in DEVICES.keys()), 2
        )
        records.append(row)

    df = pd.DataFrame(records)
    return df

# ── Run & Save ─────────────────────────────────────────
if __name__ == "__main__":
    print("⚡ Simulating smart home energy data...")
    df = simulate_usage()

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/energy_data.csv", index=False)

    print(f"✅ Done! {len(df)} hourly records saved to data/energy_data.csv")
    print(df.head(10))
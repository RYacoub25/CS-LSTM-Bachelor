import pandas as pd
import numpy as np

# Load files
ego_df = pd.read_feather("G:/My Drive/Argoverse 2 Dataset/Sensor Dataset/Validation Part 1/sensor/val/2ff4f798-78d9-3384-87e9-61928aa4cb6d/city_SE3_egovehicle.feather")
ann_df = pd.read_feather("G:/My Drive/Argoverse 2 Dataset/Sensor Dataset/Validation Part 1/sensor/val/2ff4f798-78d9-3384-87e9-61928aa4cb6d/annotations.feather")

# Columns used for position comparison
ego_cols = ['timestamp_ns', 'tx_m', 'ty_m']
obj_cols = ['category', 'tx_m', 'ty_m', 'timestamp_ns']

# Filter out objects with known categories only (optional)
target_classes = [
    "STOP_SIGN", "SIGN", "TRAFFIC_LIGHT_TRAILER", "CONSTRUCTION_CONE",
    "CONSTRUCTION_BARREL", "MESSAGE_BOARD_TRAILER", "MOBILE_PEDESTRIAN_SIGN"
]
ann_df = ann_df[ann_df["category"].isin(target_classes)]

# Sample the first N ego timestamps
N = 10
sampled = ego_df.head(N)

for idx, row in sampled.iterrows():
    ts, ex, ey = row["timestamp_ns"], row["tx_m"], row["ty_m"]
    # Get all objects at the same timestamp (optional: remove this filter for all-time objects)
    objs = ann_df.copy()
    
    # Compute distances
    dx = objs["tx_m"] - ex
    dy = objs["ty_m"] - ey
    dists = np.sqrt(dx**2 + dy**2)
    
    min_dist = dists.min() if not dists.empty else None
    closest_idx = dists.idxmin() if not dists.empty else None

    print(f"\n📍 Timestamp: {ts}")
    print(f"Ego Location: ({ex:.2f}, {ey:.2f})")
    if min_dist is not None:
        print(f"🔎 Closest object: {objs.loc[closest_idx, 'category']} at {min_dist:.2f} meters")
    else:
        print("❌ No objects found.")

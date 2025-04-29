# social_vehicles_yaw.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Paths
input_path = "data/processed/social_vehicles.csv"
output_path = "data/processed/social_vehicles_yaw.csv"

# Load
df = pd.read_csv(input_path)

# Initialize yaw column
df['yaw'] = 0.0

# Sort first (very important)
df = df.sort_values(['track_uuid', 'timestamp_ns'])

# Compute yaw per track
for track_id, group in tqdm(df.groupby('track_uuid'), desc="Processing tracks"):
    if len(group) < 2:
        # If only 1 point → yaw = 0 (default)
        continue

    x = group['x'].values
    y = group['y'].values

    dx = np.gradient(x)
    dy = np.gradient(y)

    yaw = np.arctan2(dy, dx)  # yaw = atan2(dy/dt, dx/dt)

    # Save back
    df.loc[group.index, 'yaw'] = yaw

# Save the new CSV
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Done! Saved social vehicles with yaw to: {output_path}")

# Step 1: extract_ego_with_yaw.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from config import SENSOR_DIR, PROCESSED_DATA_DIR

# Compute velocity and acceleration
def compute_velocity_and_acceleration(df):
    df = df.sort_values('timestamp_ns')
    df['vx'] = df['x'].diff() / df['timestamp_ns'].diff()
    df['vy'] = df['y'].diff() / df['timestamp_ns'].diff()
    df['vz'] = df['z'].diff() / df['timestamp_ns'].diff()
    df['ax'] = df['vx'].diff() / df['timestamp_ns'].diff()
    df['ay'] = df['vy'].diff() / df['timestamp_ns'].diff()
    df['az'] = df['vz'].diff() / df['timestamp_ns'].diff()
    df.fillna(0, inplace=True)
    return df

# Extract ego vehicle pose with yaw
def extract_ego_vehicle_features(file):
    df = pd.read_feather(file)

    # Rename position fields
    df = df.rename(columns={"tx_m": "x", "ty_m": "y", "tz_m": "z"})

    # Extract yaw (already present in city_SE3_egovehicle.feather)
    if 'yaw' not in df.columns:
        # Calculate yaw manually from quaternion (only if missing)
        qw = df['qw']
        qx = df['qx']
        qy = df['qy']
        qz = df['qz']
        # Yaw from quaternion formula
        df['yaw'] = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))

    df = compute_velocity_and_acceleration(df)

    return df[['timestamp_ns', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'yaw']]

def extract():
    ego_data = []

    for root, _, files in tqdm(os.walk(SENSOR_DIR)):
        for file in files:
            path = os.path.join(root, file)
            if file == 'city_SE3_egovehicle.feather':
                print(f"📂 Found ego vehicle file: {path}")
                ego_data.append(extract_ego_vehicle_features(path))

    if ego_data:
        full_df = pd.concat(ego_data)
        output_path = os.path.join(PROCESSED_DATA_DIR, 'ego_vehicle_yaw.csv')
        full_df.to_csv(output_path, index=False)
        print(f"✅ Saved updated ego_vehicle.csv with yaw to {output_path}")
    else:
        print("❌ No ego vehicle data found.")

if __name__ == "__main__":
    extract()

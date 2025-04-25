import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from config import PROCESSED_DATA_DIR, SENSOR_DIR

map_classes = [
    "STOP_SIGN", "SIGN", "CONSTRUCTION_CONE", "CONSTRUCTION_BARREL",
    "MOBILE_PEDESTRIAN_SIGN", "TRAFFIC_LIGHT_TRAILER", "MESSAGE_BOARD_TRAILER"
]

def compute_velocity_and_acceleration(df):
    df = df.sort_values('timestamp_ns')
    dt = df['timestamp_ns'].diff().replace(0, np.nan)
    df['vx'] = df['x'].diff() / dt
    df['vy'] = df['y'].diff() / dt
    df['vz'] = df['z'].diff() / dt
    df['ax'] = df['vx'].diff() / dt
    df['ay'] = df['vy'].diff() / dt
    df['az'] = df['vz'].diff() / dt
    return df.fillna(0)

def extract_lane_information(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    lanes_info = []
    for lane_id, lane in data.get('lane_segments', {}).items():
        lanes_info.append({
            'lane_id': lane_id,
            'lane_type': lane.get('lane_type', 'UNKNOWN'),
            'left_lane_boundary': lane.get('left_lane_boundary', []),
            'right_lane_boundary': lane.get('right_lane_boundary', []),
            'left_lane_mark_type': lane.get('left_lane_mark_type', 'UNKNOWN'),
            'right_lane_mark_type': lane.get('right_lane_mark_type', 'UNKNOWN')
        })
    return lanes_info

def extract_topography(npy_file):
    try:
        data = np.load(npy_file)
        return {'ground_height': float(data.mean()) if data.size > 0 else np.nan}
    except Exception as e:
        print(f"❌ Failed topography file {npy_file}: {e}")
        return {'ground_height': np.nan}

def extract_all():
    ego_data, social_data, map_data, constant_data = [], [], [], []

    for root, _, files in tqdm(os.walk(SENSOR_DIR)):
        ego_file, social_file = None, None

        for file in files:
            if file == 'city_SE3_egovehicle.feather':
                ego_file = os.path.join(root, file)
            elif file == 'annotations.feather':
                social_file = os.path.join(root, file)
            elif file.endswith('.json'):
                constant_data.extend(extract_lane_information(os.path.join(root, file)))
            elif file.endswith('.npy'):
                constant_data.append(extract_topography(os.path.join(root, file)))

        if ego_file and social_file:
            ego_df = pd.read_feather(ego_file)
            ego_df = ego_df.rename(columns={"tx_m": "x", "ty_m": "y", "tz_m": "z"})
            ego_df = compute_velocity_and_acceleration(ego_df)
            ego_data.append(ego_df)

            social_df = pd.read_feather(social_file)
            merged = social_df.merge(ego_df[['timestamp_ns', 'x', 'y', 'z']], on='timestamp_ns', how='inner')
            merged['vx'] = merged['x'].diff()
            merged['vy'] = merged['y'].diff()
            merged['vz'] = merged['z'].diff()
            merged['ax'] = merged['vx'].diff()
            merged['ay'] = merged['vy'].diff()
            merged['az'] = merged['vz'].diff()
            social_data.append(merged[['timestamp_ns', 'track_uuid', 'category', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']])

            # Filter map-related objects
            map_objs = social_df[social_df['category'].isin(map_classes)]
            merged_map = map_objs.merge(ego_df[['timestamp_ns', 'x', 'y', 'z']], on='timestamp_ns', how='inner')
            merged_map['x'] = merged_map['tx_m'] + merged_map['x']
            merged_map['y'] = merged_map['ty_m'] + merged_map['y']
            merged_map['z'] = merged_map['tz_m'] + merged_map['z']
            map_data.append(merged_map[['timestamp_ns', 'category', 'x', 'y', 'z']])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    if ego_data:
        pd.concat(ego_data).to_csv(os.path.join(PROCESSED_DATA_DIR, 'ego_vehicle.csv'), index=False)
    if social_data:
        pd.concat(social_data).to_csv(os.path.join(PROCESSED_DATA_DIR, 'social_vehicles.csv'), index=False)
    if map_data:
        pd.concat(map_data).to_csv(os.path.join(PROCESSED_DATA_DIR, 'map_objects.csv'), index=False)
    if constant_data:
        pd.DataFrame(constant_data).to_csv(os.path.join(PROCESSED_DATA_DIR, 'constant_features.csv'), index=False)

    print("✅ Extraction and saving completed.")


if __name__ == "__main__":
    extract_all()

    print("✅ All done!")
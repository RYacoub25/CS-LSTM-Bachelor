# dataset.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

def load_and_preprocess(ego_path, social_path):
    # Load data
    ego = pd.read_csv(ego_path)
    social = pd.read_csv(social_path)

    features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    
    # Check for NaNs before scaling
    if ego[features].isnull().any().any():
        print(f"NaN values detected in ego vehicle data in columns: {ego[features].isnull().sum()}")
        raise ValueError("NaN values detected in ego vehicle data")
    
    if social[features].isnull().any().any():
        print(f"NaN values detected in social vehicle data in columns: {social[features].isnull().sum()}")
        # Drop rows with NaN values
        social.dropna(subset=features, inplace=True)
        print(f"Rows with NaN values in social vehicle data have been dropped.")
    
    # Perform scaling
    scaler = StandardScaler()
    ego[features] = scaler.fit_transform(ego[features])
    social[features] = scaler.transform(social[features])
    
    return ego, social

def create_sequences_with_neighbors(ego_df, social_df, seq_len=30, target_step=30, grid_size=3.0, max_neighbors=10):
    timestamps = ego_df['timestamp_ns'].values
    X_hist, X_nbrs, masks, y_targets = [], [], [], []

    for i in range(len(ego_df) - seq_len - target_step):
        ego_hist = ego_df.iloc[i:i+seq_len][['x','y','z','vx','vy','vz','ax','ay','az']].values
        target = ego_df.iloc[i+seq_len+target_step-1][['x', 'y']].values
        ego_pos = ego_df.iloc[i+seq_len-1][['x', 'y']].values
        timestamp = ego_df.iloc[i+seq_len-1]['timestamp_ns']

        # Grid boundaries
        xmin, xmax = ego_pos[0] - grid_size, ego_pos[0] + grid_size
        ymin, ymax = ego_pos[1] - grid_size, ego_pos[1] + grid_size

        # Find effective neighbors (same grid)
        neighbors = social_df[
            (social_df['timestamp_ns'] == timestamp) &
            (social_df['x'] >= xmin) & (social_df['x'] <= xmax) &
            (social_df['y'] >= ymin) & (social_df['y'] <= ymax)
        ].sort_values('track_uuid')

        # Check for NaNs in neighbors
        if neighbors[['x','y','z','vx','vy','vz','ax','ay','az']].isnull().any().any():
            print(f"Warning: NaN detected in neighbors at timestamp {timestamp}")
            continue  # Skip this batch

        nbr_feats = neighbors[['x','y','z','vx','vy','vz','ax','ay','az']].values[:max_neighbors]
        pad_count = max_neighbors - len(nbr_feats)
        if pad_count > 0:
            nbr_feats = np.vstack([nbr_feats, np.zeros((pad_count, 9))])

        mask = np.zeros((max_neighbors, 1))
        mask[:len(neighbors[:max_neighbors])] = 1

        X_hist.append(ego_hist)
        X_nbrs.append(nbr_feats)
        masks.append(mask)
        y_targets.append(target)

    return np.array(X_hist), np.array(X_nbrs), np.array(masks), np.array(y_targets)

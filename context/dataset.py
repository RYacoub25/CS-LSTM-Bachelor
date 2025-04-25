import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Function to load and preprocess ego and social vehicle data
def load_and_preprocess(ego_path, social_path):
    ego = pd.read_csv(ego_path)
    social = pd.read_csv(social_path)

    features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    scaler = StandardScaler()
    ego[features] = scaler.fit_transform(ego[features])
    social[features] = scaler.transform(social[features])
    
    return ego, social

# Function to load and preprocess contextual features (road geometry and lane info)
def load_contextual_features(context_path):
    # Load the contextual features (like ground height, lane ID, etc.)
    context_df = pd.read_csv(context_path)
    
    # You can add preprocessing steps here if needed (e.g., encoding categorical variables)
    context_df = context_df[['ground_height', 'lane_id', 'lane_type']]  # Keep only the relevant features
    return context_df

# Function to create sequences of historical data and neighbors' data
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
        neighbors = social_df[(
            social_df['timestamp_ns'] == timestamp) &
            (social_df['x'] >= xmin) & (social_df['x'] <= xmax) &
            (social_df['y'] >= ymin) & (social_df['y'] <= ymax)
        ].sort_values('track_uuid')

        # Extract features of neighbors
        nbr_feats = neighbors[['x','y','z','vx','vy','vz','ax','ay','az']].values[:max_neighbors]
        pad_count = max_neighbors - len(nbr_feats)
        if pad_count > 0:
            nbr_feats = np.vstack([nbr_feats, np.zeros((pad_count, 9))])  # Padding neighbors to max_neighbors

        mask = np.zeros((max_neighbors, 1))
        mask[:len(neighbors[:max_neighbors])] = 1  # Mask valid neighbors

        # Append to lists
        X_hist.append(ego_hist)
        X_nbrs.append(nbr_feats)
        masks.append(mask)
        y_targets.append(target)

    return np.array(X_hist), np.array(X_nbrs), np.array(masks), np.array(y_targets)

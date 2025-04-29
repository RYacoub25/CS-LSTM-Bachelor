import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess ego vehicle data
def load_ego_data(ego_path):
    # Load ego vehicle data (e.g., CSV file)
    ego_df = pd.read_csv(ego_path)
    
    # Assuming ego_df contains the data for ego vehicle
    # Extract relevant features for scaling (e.g., x, y, z, velocity, etc.)
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    scaler = StandardScaler()
    ego_df[features] = scaler.fit_transform(ego_df[features])  # Scale the ego data
    
    # Compute `delta_yaw` (yaw angle change between consecutive frames)
    ego_df['delta_yaw'] = ego_df['yaw'].diff()  # Assuming 'yaw' column exists
    
    # Define threshold for turn vs. straight
    threshold = 0.1  # This can be adjusted based on your data
    
    # Classify intention based on delta_yaw
    ego_df['intention'] = np.where(ego_df['delta_yaw'].abs() < threshold, 0,  # 0 -> Straight
                                    np.where(ego_df['delta_yaw'] > 0, 1, -1))  # 1 -> Left Turn, -1 -> Right Turn
    
    return ego_df


# Function to load and preprocess social vehicle data
def load_social_data(social_path):
    # Load social vehicle data (e.g., CSV file)
    social_df = pd.read_csv(social_path)
    
    # Assuming social_df contains the data for social vehicles (neighbors)
    # Extract features for scaling
    features = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
    scaler = StandardScaler()
    social_df[features] = scaler.fit_transform(social_df[features])  # Scale the social vehicle data

    return social_df


# Function to load and preprocess contextual data (merged lanes and map objects)
def load_contextual_data(contextual_path):
    # Load the merged contextual data (lanes and map objects)
    contextual_data = np.load(contextual_path)  # Shape: (samples, features)
    
    # Filter out samples with NaN values
    valid_samples = ~np.isnan(contextual_data).any(axis=1)  # Check for NaNs across all features for each sample
    contextual_clean = contextual_data[valid_samples]
    
    return contextual_clean


# Function to load and preprocess all data
# Function to load and preprocess all data
def load_and_preprocess(ego_path, social_path, contextual_path):
    # Load and preprocess each type of data
    ego_df = load_ego_data(ego_path)
    social_df = load_social_data(social_path)
    combined_contextual = load_contextual_data(contextual_path)
    
    # Create sequences with neighbors for training (we will apply the NaN mask here)
    X_hist, X_nbrs, X_mask, y = create_sequences_with_neighbors(ego_df, social_df, seq_len=30, target_step=30)
    
    # Extract intention as a separate feature (as part of the dataset)
    intention = ego_df['intention'].values  # Extract intention (turn vs straight)
    
    # Step 1: Align lengths of `X_nbrs` and `contextual` to ensure they are the same length
    min_len = min(len(X_nbrs), len(combined_contextual), len(intention))  # Get the smaller length
    X_nbrs = X_nbrs[:min_len]
    combined_contextual = combined_contextual[:min_len]
    intention = intention[:min_len]  # Ensure intention is the same length
    
    # Step 2: Apply NaN masking for both X_nbrs and contextual to ensure only valid samples
    valid_samples_X_nbrs = ~np.isnan(X_nbrs).any(axis=(1, 2))  # Check for NaNs across neighbors and features
    valid_samples_contextual = ~np.isnan(combined_contextual).any(axis=1)  # Check for NaNs in contextual features
    
    # Step 3: Combine both masks to filter the data
    final_mask = valid_samples_X_nbrs & valid_samples_contextual
    
    # Filter out the invalid samples based on the combined mask
    X_hist_clean = X_hist[final_mask]
    X_nbrs_clean = X_nbrs[final_mask]
    X_mask_clean = X_mask[final_mask]
    y_clean = y[final_mask]
    contextual_clean = combined_contextual[final_mask]  # Apply final mask to contextual data
    intention_clean = intention[final_mask]  # Apply final mask to intention data
    
    return X_hist_clean, X_nbrs_clean, X_mask_clean, y_clean, contextual_clean, intention_clean

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

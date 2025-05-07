
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_ego_data(ego_path):
    ego_df = pd.read_csv(ego_path)
    features = ['x','y','z','vx','vy','vz','ax','ay','az']
    scaler = StandardScaler()
    ego_df[features] = scaler.fit_transform(ego_df[features])
    ego_df['delta_yaw'] = ego_df['yaw'].diff()
    threshold = 0.1
    ego_df['intention'] = np.where(
        ego_df['delta_yaw'].abs() < threshold, 0,
        np.where(ego_df['delta_yaw'] > 0, 1, -1)
    )
    return ego_df

def load_social_data(social_path):
    social_df = pd.read_csv(social_path)
    features = ['x','y','z','vx','vy','vz','ax','ay','az']
    scaler = StandardScaler()
    social_df[features] = scaler.fit_transform(social_df[features])
    return social_df

def load_contextual_data(contextual_path):
    data = np.load(contextual_path)
    valid = ~np.isnan(data).any(axis=1)
    return data[valid]

def create_sequences_with_neighbors(
    ego_df, social_df,
    seq_len=30, target_step=30,
    grid_size=3.0, max_neighbors=10
):
    X_hist, X_nbrs, X_mask, y_targets = [], [], [], []
    X_nbr_pos, X_ego_pos = [], []

    for i in range(len(ego_df) - seq_len - target_step):
        # historical ego trajectory
        ego_hist = ego_df.iloc[i:i+seq_len][
            ['x','y','z','vx','vy','vz','ax','ay','az']
        ].values.astype(np.float32)

        # prediction target (x,y)
        target = ego_df.iloc[i+seq_len+target_step-1][['x','y']].values.astype(np.float32)

        # current ego position
        ego_pos = ego_df.iloc[i+seq_len-1][['x','y']].values.astype(np.float32)

        # grid window
        xmin, xmax = ego_pos[0]-grid_size, ego_pos[0]+grid_size
        ymin, ymax = ego_pos[1]-grid_size, ego_pos[1]+grid_size

        # select neighbors at current timestamp
        ts = ego_df.iloc[i+seq_len-1]['timestamp_ns']
        neighs = social_df[
            (social_df['timestamp_ns']==ts) &
            (social_df['x'].between(xmin,xmax)) &
            (social_df['y'].between(ymin,ymax))
        ].sort_values('track_uuid')

        # neighbor features
        feats = neighs[['x','y','z','vx','vy','vz','ax','ay','az']].values.astype(np.float32)[:max_neighbors]
        nbr_pos = neighs[['x','y']].values.astype(np.float32)[:max_neighbors]
        pad = max_neighbors - len(feats)
        if pad>0:
            feats = np.vstack([feats, np.zeros((pad,9),dtype=np.float32)])
            nbr_pos = np.vstack([nbr_pos, np.zeros((pad,2),dtype=np.float32)])

        mask = np.zeros((max_neighbors,1), dtype=np.float32)
        mask[:len(neighs[:max_neighbors])] = 1.0

        # collect
        X_hist.append(ego_hist)
        X_nbrs.append(feats)
        X_mask.append(mask)
        y_targets.append(target)
        X_nbr_pos.append(nbr_pos)
        X_ego_pos.append(ego_pos)

    return (
        np.array(X_hist), np.array(X_nbrs), np.array(X_mask),
        np.array(y_targets), np.array(X_nbr_pos), np.array(X_ego_pos)
    )

def load_and_preprocess(ego_path, social_path, contextual_path):
    ego_df    = load_ego_data(ego_path)
    social_df = load_social_data(social_path)
    ctx       = load_contextual_data(contextual_path)

    X_hist, X_nbrs, X_mask, y, nbr_pos, ego_pos = create_sequences_with_neighbors(
        ego_df, social_df, seq_len=30, target_step=30
    )
    intention = ego_df['intention'].values.astype(np.float32)

    # align all arrays to same length
    min_len = min(
        len(X_hist), len(X_nbrs), len(X_mask),
        len(y),      len(ctx),   len(intention),
        len(nbr_pos), len(ego_pos)
    )
    X_hist    = X_hist[:min_len]
    X_nbrs    = X_nbrs[:min_len]
    X_mask    = X_mask[:min_len]
    y         = y[:min_len]
    ctx       = ctx[:min_len]
    intention = intention[:min_len]
    nbr_pos   = nbr_pos[:min_len]
    ego_pos   = ego_pos[:min_len]

    # filter out NaNs
    valid_nbr = ~np.isnan(X_nbrs).any(axis=(1,2))
    valid_ctx = ~np.isnan(ctx).any(axis=1)
    mask_all  = valid_nbr & valid_ctx

    return (
      X_hist[mask_all],
      X_nbrs[mask_all],
      X_mask[mask_all],
      y[mask_all],
      ctx[mask_all],
      intention[mask_all],
      nbr_pos[mask_all],
      ego_pos[mask_all],
    )

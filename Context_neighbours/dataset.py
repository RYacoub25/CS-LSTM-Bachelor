import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_ego_data(ego_path):
    ego_df = pd.read_csv(ego_path)
    feats = ['x','y','z','vx','vy','vz','ax','ay','az']
    scaler = StandardScaler()
    ego_df[feats] = scaler.fit_transform(ego_df[feats])
    ego_df['delta_yaw'] = ego_df['yaw'].diff()
    thr = 0.1
    ego_df['intention'] = np.where(
        ego_df['delta_yaw'].abs() < thr, 0,
        np.where(ego_df['delta_yaw'] > 0, 1, -1)
    )
    return ego_df

def load_social_data(social_path):
    social_df = pd.read_csv(social_path)
    feats = ['x','y','z','vx','vy','vz','ax','ay','az']
    scaler = StandardScaler()
    social_df[feats] = scaler.fit_transform(social_df[feats])
    return social_df

def load_contextual_data(contextual_path):
    ctx = np.load(contextual_path)
    return ctx[~np.isnan(ctx).any(axis=1)]

def create_sequences_with_neighbors(
    ego_df, social_df,
    seq_len=30, target_step=30,
    grid_size=3.0, max_neighbors=10,
    radius=None
):
    X_hist, X_nbrs, X_mask, Y, intents = [], [], [], [], []
    for i in range(len(ego_df) - seq_len - target_step):
        window = ego_df.iloc[i:i+seq_len]
        ego_hist = window[['x','y','z','vx','vy','vz','ax','ay','az']].values
        target   = ego_df.iloc[i+seq_len+target_step-1][['x','y']].values
        ego_pos  = ego_df.iloc[i+seq_len-1][['x','y']].values
        ts       = ego_df.iloc[i+seq_len-1]['timestamp_ns']

        # grid filtering
        xmin, xmax = ego_pos[0]-grid_size, ego_pos[0]+grid_size
        ymin, ymax = ego_pos[1]-grid_size, ego_pos[1]+grid_size
        nbrs = social_df[
            (social_df['timestamp_ns']==ts) &
            social_df['x'].between(xmin,xmax) &
            social_df['y'].between(ymin,ymax)
        ].sort_values('track_uuid')

        feats = nbrs[['x','y','z','vx','vy','vz','ax','ay','az']].values[:max_neighbors]
        pad   = max_neighbors - len(feats)
        if pad>0:
            feats = np.vstack([feats, np.zeros((pad,9))])

        mask = np.zeros((max_neighbors,1))
        mask[:len(nbrs[:max_neighbors])] = 1

        # radius filtering (in the SAME scaled units):
        if radius is not None:
            dists = np.linalg.norm(feats[:len(nbrs[:max_neighbors]),:2] - ego_pos, axis=1)
            for j, d in enumerate(dists):
                if d > radius:
                    feats[j] = 0
                    mask[j]  = 0

        X_hist.append(ego_hist)
        X_nbrs.append(feats)
        X_mask.append(mask)
        Y.append(target)
        intents.append(
            ego_df.iloc[i+seq_len-1]['intention']
        )

    return (
        np.array(X_hist),
        np.array(X_nbrs),
        np.array(X_mask),
        np.array(Y),
        np.array(intents)
    )

def load_and_preprocess(
    ego_path, social_path, contextual_path,
    seq_len=30, target_step=30,
    grid_size=3.0, max_neighbors=10,
    radius=5.0
):
    ego_df    = load_ego_data(ego_path)
    social_df = load_social_data(social_path)
    ctx       = load_contextual_data(contextual_path)

    X_hist, X_nbrs, X_mask, Y, intents = create_sequences_with_neighbors(
        ego_df, social_df,
        seq_len, target_step,
        grid_size, max_neighbors,
        radius=radius
    )

    # align with context length
    min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(Y), len(ctx), len(intents))
    X_hist, X_nbrs, X_mask, Y = (
        X_hist[:min_len],
        X_nbrs[:min_len],
        X_mask[:min_len],
        Y[:min_len],
    )
    ctx     = ctx[:min_len]
    intents = intents[:min_len]

    # filter out any rows with NaNs
    valid_nbr = ~np.isnan(X_nbrs).any(axis=(1,2))
    valid_ctx = ~np.isnan(ctx).any(axis=1)
    keep      = valid_nbr & valid_ctx

    return (
        X_hist[keep],
        X_nbrs[keep],
        X_mask[keep],
        Y[keep],
        ctx[keep],
        intents[keep]
    )

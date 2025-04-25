import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm

# Paths to preprocessed files
EGO_PATH = "data/processed/ego_vehicle.csv"
SOCIAL_PATH = "data/processed/social_vehicles.csv"
LANE_PATH = "data/processed/constant_features.csv"
OBJECTS_PATH = "data/processed/map_objects.csv"
OUTPUT_VIDEO = "visual/output_scene.mp4"

# Load all data
ego_df = pd.read_csv(EGO_PATH)
social_df = pd.read_csv(SOCIAL_PATH)
lane_df = pd.read_csv(LANE_PATH)
obj_df = pd.read_csv(OBJECTS_PATH)

# Focus area radius
RADIUS = 50  # meters
FRAMES = 100  # number of timestamps
START_IDX = 1000  # starting index

# Get sequence of timestamps from ego
timestamps = ego_df["timestamp_ns"].unique()
timestamps = timestamps[START_IDX:START_IDX + FRAMES]

# Plotting function for each frame
def plot_frame(i):
    ts = timestamps[i]
    plt.clf()

    ego = ego_df[ego_df["timestamp_ns"] == ts]
    if ego.empty:
        return

    ex, ey = ego.iloc[0][["x", "y"]]

    # Ego
    plt.scatter(ex, ey, c='red', label="Ego Vehicle", zorder=5)

    # Social vehicles
    local_social = social_df[social_df["timestamp_ns"] == ts]
    local_social = local_social[
        ((local_social["x"] - ex) ** 2 + (local_social["y"] - ey) ** 2) < RADIUS ** 2
    ]
    plt.scatter(local_social["x"], local_social["y"], c='blue', label="Social Vehicles", s=10)

    # Lane boundaries
    for _, row in lane_df.iterrows():
        try:
            left = eval(row["left_lane_boundary"])
            right = eval(row["right_lane_boundary"])
            left = np.array([[p["x"], p["y"]] for p in left])
            right = np.array([[p["x"], p["y"]] for p in right])
            if np.linalg.norm(left.mean(axis=0) - [ex, ey]) < RADIUS:
                plt.plot(left[:, 0], left[:, 1], c='green', linewidth=1)
            if np.linalg.norm(right.mean(axis=0) - [ex, ey]) < RADIUS:
                plt.plot(right[:, 0], right[:, 1], c='black', linewidth=1)
        except:
            continue

    # Map objects
    obj_subset = obj_df[((obj_df["tx_m"] - ex) ** 2 + (obj_df["ty_m"] - ey) ** 2) < RADIUS ** 2]
    plt.scatter(obj_subset["tx_m"], obj_subset["ty_m"], c='purple', marker='x', label="Map Objects", s=20)

    plt.title(f"Scene at timestamp {ts}")
    plt.xlim(ex - RADIUS, ex + RADIUS)
    plt.ylim(ey - RADIUS, ey + RADIUS)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper right")
    plt.grid(True)

# Create animation
fig = plt.figure(figsize=(8, 6))
ani = FuncAnimation(fig, plot_frame, frames=FRAMES, interval=200)

# Save to video
writer = FFMpegWriter(fps=5)
os.makedirs("visual", exist_ok=True)
ani.save(OUTPUT_VIDEO, writer=writer)

import pathlib
pathlib.Path(OUTPUT_VIDEO)

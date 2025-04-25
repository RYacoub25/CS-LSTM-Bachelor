import pandas as pd
import os

# Paths
input_path = 'data/processed/social_vehicles.csv'
output_path = 'data/processed/social_vehicles.csv'

# Check the file exists
if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ File not found: {input_path}")

# Load the social context data
df = pd.read_csv(input_path)

# Sort for proper diff calculations
df = df.sort_values(by=['track_uuid', 'timestamp_ns'])

# Compute velocities
df['vx'] = df.groupby('track_uuid')['global_x'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
df['vy'] = df.groupby('track_uuid')['global_y'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
df['vz'] = df.groupby('track_uuid')['global_z'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()

# Compute accelerations
df['ax'] = df.groupby('track_uuid')['vx'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
df['ay'] = df.groupby('track_uuid')['vy'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
df['az'] = df.groupby('track_uuid')['vz'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()

# Save the updated DataFrame
df.to_csv(output_path, index=False)
print(f"✅ Saved updated social features to: {output_path}")

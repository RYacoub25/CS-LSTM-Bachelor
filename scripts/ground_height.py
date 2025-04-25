import os
import pandas as pd
import numpy as np
from config import PROCESSED_DATA_DIR, SENSOR_DIR

def find_ground_height_from_npy(sensor_dir):
    valid_heights = []
    for root, _, files in os.walk(sensor_dir):
        for file in files:
            if file.endswith('.npy') and 'ground_height' in file:
                path = os.path.join(root, file)
                try:
                    data = np.load(path)
                    if np.issubdtype(data.dtype, np.number) and data.size > 0:
                        mean_height = float(np.nanmean(data))
                        if np.isfinite(mean_height) and -1000 < mean_height < 1000:
                            valid_heights.append(mean_height)
                except Exception as e:
                    print(f"❌ Error reading {file}: {e}")
    if valid_heights:
        return sum(valid_heights) / len(valid_heights)
    return 0.0  # fallback


def fix_ground_height():
    csv_path = os.path.join(PROCESSED_DATA_DIR, 'constant_features.csv')

    if not os.path.exists(csv_path):
        print("❌ constant_features.csv not found.")
        return

    df = pd.read_csv(csv_path)

    if 'ground_height' not in df.columns or df['ground_height'].isna().all():
        print("⚠️  Missing or empty ground_height column. Attempting to fix...")

        ground_height = find_ground_height_from_npy(SENSOR_DIR)
        if ground_height is not None:
            df['ground_height'] = ground_height
            df.to_csv(csv_path, index=False)
            print(f"✅ ground_height filled with value: {ground_height:.3f}")
        else:
            print("❌ Could not extract ground height from .npy files.")
    else:
        print("✅ ground_height already populated.")

if __name__ == '__main__':
    fix_ground_height()

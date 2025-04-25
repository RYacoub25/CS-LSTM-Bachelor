import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from context.model import ContextualSocialLSTM  # or SocialLSTM if you have it separately
from social_lstm.dataset import load_and_preprocess, create_sequences_with_neighbors
from sklearn.model_selection import train_test_split

# --- Config ---
MODEL_PATH = "contextual_social_lstm.pth"  # or "social_lstm.pth"
OUTPUT_CSV = "contextual_preds.csv"        # or "social_preds.csv"
CONTEXTUAL = True                          # False for Social LSTM

# --- Load data ---
ego_df, social_df = load_and_preprocess('data/processed/ego_vehicle.csv', 'data/processed/social_vehicles.csv')
X_hist, X_nbrs, X_mask, y = create_sequences_with_neighbors(ego_df, social_df)
context = np.load("data/processed/contextual_features.npy")

# Align
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(context))
X_hist, X_nbrs, X_mask, y, context = map(lambda x: x[:min_len], [X_hist, X_nbrs, X_mask, y, context])

# Remove NaNs
valid = ~np.isnan(X_nbrs).any(axis=(1, 2)) & ~np.isnan(context).any(axis=1)
X_hist, X_nbrs, X_mask, y, context = X_hist[valid], X_nbrs[valid], X_mask[valid], y[valid], context[valid]

# Split
X_hist_train, X_hist_test, y_train, y_test, ctx_train, ctx_test, Xn_train, Xn_test, Xm_train, Xm_test = train_test_split(
    X_hist, y, context, X_nbrs, X_mask, test_size=0.2, random_state=42
)


# Dataset
test_ds = TensorDataset(
    torch.tensor(X_hist_test, dtype=torch.float32),
    torch.tensor(Xn_test, dtype=torch.float32),
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(ctx_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_dl = DataLoader(test_ds, batch_size=64)

# Model
model = ContextualSocialLSTM(context_dim=ctx_test.shape[1]) if CONTEXTUAL else ContextualSocialLSTM(context_dim=0)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Inference
rows = []
with torch.no_grad():
    for hist, nbrs, mask, ctx, target in test_dl:
        context_input = ctx if CONTEXTUAL else torch.zeros_like(ctx[:, :1])
        pred = model(hist, nbrs, mask, context_input)
        for i in range(len(target)):
            rows.append({
                "x_gt": target[i, 0].item(),
                "y_gt": target[i, 1].item(),
                "x_pred": pred[i, 0].item(),
                "y_pred": pred[i, 1].item()
            })

# Save
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to: {OUTPUT_CSV}")

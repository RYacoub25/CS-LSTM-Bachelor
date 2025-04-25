# train.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess, create_sequences_with_neighbors
from model import ContextualSocialLSTM, apply_weight_init
import numpy as np
import pandas as pd
from torch import nn

# Load ego and social vehicle data
ego_df, social_df = load_and_preprocess('data/processed/ego_vehicle.csv', 'data/processed/social_vehicles.csv')
X_hist, X_nbrs, X_mask, y = create_sequences_with_neighbors(ego_df, social_df)

# Load both context features
#lanes = np.load("data/processed/contextual_features.npy")              # shape: (N, lane_dim)
#objects = np.load("data/processed/contextual_objects.npy")             # shape: (N, object_dim)

# Combine
#combined_context = np.concatenate([lanes, objects], axis=1)            # shape: (N, lane_dim + object_dim)

# Use in train.py
#contextual = combined_context
contextual = np.load("data/processed/contextual_features_merged.npy")
print("context.shape =", contextual.shape)
print("X_hist.shape  =", X_hist.shape)
print("y.shape       =", y.shape)

# Align all inputs to the same length
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(contextual))
X_hist = X_hist[:min_len]
X_nbrs = X_nbrs[:min_len]
X_mask = X_mask[:min_len]
y = y[:min_len]
contextual = contextual[:min_len]


# Remove samples with NaNs in neighbors or context
mask_valid = ~np.isnan(X_nbrs).any(axis=(1, 2)) & ~np.isnan(contextual).any(axis=1)
X_hist = X_hist[mask_valid]
X_nbrs = X_nbrs[mask_valid]
X_mask = X_mask[mask_valid]
y = y[mask_valid]
contextual = contextual[mask_valid]


# Train-test split
X_hist_train, X_hist_test, y_train, y_test, ctx_train, ctx_test, Xn_train, Xn_test, Xm_train, Xm_test = train_test_split(
    X_hist, y, contextual, X_nbrs, X_mask, test_size=0.2, random_state=42
)

# DataLoader
train_ds = TensorDataset(
    torch.tensor(X_hist_train, dtype=torch.float32),
    torch.tensor(Xn_train, dtype=torch.float32),
    torch.tensor(Xm_train, dtype=torch.float32),
    torch.tensor(ctx_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# Model setup
model = ContextualSocialLSTM(context_dim=contextual.shape[1])
apply_weight_init(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for hist, nbrs, mask, context, target in train_dl:
        if torch.isnan(nbrs).any() or torch.isnan(context).any():
            continue
        pred = model(hist, nbrs, mask, context)
        if torch.isnan(pred).any():
            continue
        loss = criterion(pred, target)
        if torch.isnan(loss):
            continue
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_dl):.4f}")

# Save model
torch.save(model.state_dict(), "contextual_social_lstm.pth")
print("\u2705 Model saved as 'contextual_social_lstm.pth'")

# Evaluation
test_ds = TensorDataset(
    torch.tensor(X_hist_test, dtype=torch.float32),
    torch.tensor(Xn_test, dtype=torch.float32),
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(ctx_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_dl = DataLoader(test_ds, batch_size=64)

model.eval()
ade_list, fde_list = [], []
with torch.no_grad():
    for hist, nbrs, mask, context, target in test_dl:
        pred = model(hist, nbrs, mask, context)
        if torch.isnan(pred).any():
            continue
        ade = torch.norm(pred - target, dim=1).mean()
        fde = torch.norm(pred - target, dim=1).mean()
        ade_list.append(ade.item())
        fde_list.append(fde.item())

print(f"\u2705 Contextual Social LSTM | ADE: {sum(ade_list)/len(ade_list):.4f} | FDE: {sum(fde_list)/len(fde_list):.4f}")

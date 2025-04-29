import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess
import numpy as np
import pandas as pd
from model import ContextualSocialLSTM, apply_weight_init
from torch import nn

# Load and preprocess the data
ego_path = 'data/processed/ego_vehicle_with_intention.csv'
social_path = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'  # Path to the merged contextual features

# Get preprocessed data (cleaned and aligned)
X_hist, X_nbrs, X_mask, y, contextual, intention = load_and_preprocess(ego_path, social_path, contextual_path)

# Print the shapes of the arrays
print("context.shape =", contextual.shape)
print("X_hist.shape  =", X_hist.shape)
print("y.shape       =", y.shape)

# Step 1: Ensure y_train contains numeric values
y = np.asarray(y, dtype=np.float32)  # Ensure y is of type float32

# Step 2: Align all inputs to the same length
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(contextual), len(intention))
X_hist = X_hist[:min_len]
X_nbrs = X_nbrs[:min_len]
X_mask = X_mask[:min_len]
y = y[:min_len]
contextual = contextual[:min_len]
intention = intention[:min_len]  # Ensure intention is aligned

# Print the shapes of the arrays after trimming
print(f"X_hist.shape after alignment: {X_hist.shape}")
print(f"X_nbrs.shape after alignment: {X_nbrs.shape}")
print(f"X_mask.shape after alignment: {X_mask.shape}")
print(f"y.shape after alignment: {y.shape}")
print(f"contextual.shape after alignment: {contextual.shape}")
print(f"intention.shape after alignment: {intention.shape}")

# Step 3: Train-test split
X_hist_train, X_hist_test, y_train, y_test, ctx_train, ctx_test, Xn_train, Xn_test, Xm_train, Xm_test, intention_train, intention_test = train_test_split(
    X_hist, y, contextual, X_nbrs, X_mask, intention, test_size=0.2, random_state=42
)

# DataLoader
train_ds = TensorDataset(
    torch.tensor(X_hist_train, dtype=torch.float32),
    torch.tensor(Xn_train, dtype=torch.float32),
    torch.tensor(Xm_train, dtype=torch.float32),
    torch.tensor(ctx_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32),
    torch.tensor(intention_train, dtype=torch.float32)  # Add intention as an additional feature
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# Model setup
model = ContextualSocialLSTM(context_dim=contextual.shape[1], intention_dim=1)  # Add intention_dim as an input
apply_weight_init(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for hist, nbrs, mask, context, target, intention in train_dl:
        intention = intention.unsqueeze(1).float()  # Ensure intention is scalar and float32
        
        if torch.isnan(nbrs).any() or torch.isnan(context).any() or torch.isnan(intention).any():
            continue
        
        pred = model(hist, nbrs, mask, context, intention)  # Pass intention as the correct feature
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

# Save the trained model
torch.save(model.state_dict(), "contextual_social_lstm.pth")
print("✅ Model saved as 'contextual_social_lstm.pth'")

# Now, let's do evaluation

# Evaluation
test_ds = TensorDataset(
    torch.tensor(X_hist_test, dtype=torch.float32),
    torch.tensor(Xn_test, dtype=torch.float32),
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(ctx_test, dtype=torch.float32),
    torch.tensor(y_test[:, :2], dtype=torch.float32),  # Position as target (x, y)
    torch.tensor(intention_test, dtype=torch.float32)  # Intention as a separate input feature
)

test_dl = DataLoader(test_ds, batch_size=64)

model.eval()
ade_list, fde_list = [], []
with torch.no_grad():
    for hist, nbrs, mask, context, target_position, intention in test_dl:
        intention = intention.float()  # Convert intention to float32 (ensure the same dtype)
        
        # Pass intention to the model
        pred = model(hist, nbrs, mask, context, intention)  # Pass intention as the correct feature
        if torch.isnan(pred).any():
            continue
        
        # Calculate ADE (Average Displacement Error) and FDE (Final Displacement Error) for position
        ade = torch.norm(pred[:, :2] - target_position, dim=1).mean()  # Assuming first two columns are position (x, y)
        fde = torch.norm(pred[:, :2] - target_position, dim=1).mean()  # Same as above
        ade_list.append(ade.item())
        fde_list.append(fde.item())

# Calculate average ADE and FDE
average_ade = sum(ade_list) / len(ade_list)
average_fde = sum(fde_list) / len(fde_list)

print(f"✅ Contextual Social LSTM | ADE: {average_ade:.4f} | FDE: {average_fde:.4f}")

# train.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess, create_sequences_with_neighbors
from model import SocialLSTM
from torch import nn

# Load and preprocess the dataset
ego_df, social_df = load_and_preprocess('data/processed/ego_vehicle.csv', 'data/processed/social_vehicles.csv')
X_hist, X_nbrs, X_mask, y = create_sequences_with_neighbors(ego_df, social_df)

# Split the data into train/test sets
_, X_test, _, y_test = train_test_split(X_hist, y, test_size=0.2, random_state=42)
_, Xn_test, _, _ = train_test_split(X_nbrs, y, test_size=0.2, random_state=42)
_, Xm_test, _, _ = train_test_split(X_mask, y, test_size=0.2, random_state=42)

# Create DataLoader for test data
test_ds = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(Xn_test, dtype=torch.float32),
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)
test_dl = DataLoader(test_ds, batch_size=64)

# Initialize model, optimizer, and loss function
model = SocialLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
criterion = nn.MSELoss()

# Create DataLoader for training data
train_ds = TensorDataset(
    torch.tensor(X_hist, dtype=torch.float32),
    torch.tensor(X_nbrs, dtype=torch.float32),
    torch.tensor(X_mask, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32)
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

# Check for NaN and Inf values in data
def check_for_nans(tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in tensor of shape {tensor.shape}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in tensor of shape {tensor.shape}")

# Define the training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_idx, (hist, nbrs, mask, target) in enumerate(train_dl):
        # Check for NaN or Inf in data
        check_for_nans(hist)
        check_for_nans(nbrs)
        check_for_nans(mask)

        # Make prediction
        pred = model(hist, nbrs, mask)

        # Check for NaN or Inf in predictions
        check_for_nans(pred)

        # Calculate loss
        loss = criterion(pred, target)

        # Check for NaN in loss
        if torch.isnan(loss).any():
            print(f"NaN detected in loss at epoch {epoch}, batch {batch_idx}")
            continue  # Skip this batch

        # Backpropagation and optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_dl):.4f}")

# Save the model after training
torch.save(model.state_dict(), "social_lstm.pth")
print("✅ Model saved as 'social_lstm.pth'")

# Evaluate the model
model.eval()

ade_list, fde_list = [], []
with torch.no_grad():
    for hist, nbrs, mask, target in test_dl:
        pred = model(hist, nbrs, mask)
        ade = torch.norm(pred - target, dim=1).mean()  # Average distance error
        fde = torch.norm(pred - target, dim=1).mean()  # Final distance error
        ade_list.append(ade.item())
        fde_list.append(fde.item())

print(f"✅ Social LSTM | ADE: {sum(ade_list)/len(ade_list):.4f} | FDE: {sum(fde_list)/len(fde_list):.4f}")

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess
import numpy as np
import matplotlib.pyplot as plt
from model import ContextualSocialLSTM, apply_weight_init
from torch import nn

# ── 1) Load & preprocess ───────────────────────────────────────────────────
ego_path        = 'data/processed/ego_vehicle_with_intention.csv'
social_path     = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'

X_hist, X_nbrs, X_mask, y, contextual, intention = load_and_preprocess(
    ego_path, social_path, contextual_path
)

# align lengths and types
y = y.astype(np.float32)
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(contextual), len(intention))
X_hist    = X_hist[:min_len]
X_nbrs    = X_nbrs[:min_len]
X_mask    = X_mask[:min_len]
y         = y[:min_len]
contextual= contextual[:min_len]
intention = intention[:min_len]

# train/test split
Xh_tr,  Xh_te, \
y_tr,   y_te, \
ctx_tr, ctx_te, \
Xn_tr,  Xn_te, \
Xm_tr,  Xm_te, \
int_tr, int_te = train_test_split(
    X_hist, y, contextual, X_nbrs, X_mask, intention,
    test_size=0.2, random_state=42
)

# build DataLoaders
train_ds = TensorDataset(
    torch.tensor(Xh_tr,  dtype=torch.float32),
    torch.tensor(Xn_tr,  dtype=torch.float32),
    torch.tensor(Xm_tr,  dtype=torch.float32),
    torch.tensor(ctx_tr, dtype=torch.float32),
    torch.tensor(y_tr,   dtype=torch.float32),
    torch.tensor(int_tr, dtype=torch.float32),
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

test_ds = TensorDataset(
    torch.tensor(Xh_te,  dtype=torch.float32),
    torch.tensor(Xn_te,  dtype=torch.float32),
    torch.tensor(Xm_te,  dtype=torch.float32),
    torch.tensor(ctx_te, dtype=torch.float32),
    torch.tensor(y_te,   dtype=torch.float32),
    torch.tensor(int_te, dtype=torch.float32),
)
test_dl = DataLoader(test_ds, batch_size=64)

# ── 2) Model & training ────────────────────────────────────────────────────
model     = ContextualSocialLSTM(context_dim=contextual.shape[1], intention_dim=1)
apply_weight_init(model)
opt       = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

for epoch in range(1, 11):
    model.train()
    total_loss = 0.0
    for hist, nbrs, mask, ctx, target, intent in train_dl:
        intent = intent.unsqueeze(1)
        pred   = model(hist, nbrs, mask, ctx, intent)
        loss   = criterion(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} | Loss: {total_loss/len(train_dl):.4f}")

torch.save(model.state_dict(), "contextual_social_lstm.pth")
print("✅ Model saved")

# ── 3) Evaluation & gather all preds vs ground-truth ───────────────────────
model.eval()
all_preds, all_gt = [], []
with torch.no_grad():
    for hist, nbrs, mask, ctx, target, intent in test_dl:
        intent = intent.unsqueeze(1)
        pred   = model(hist, nbrs, mask, ctx, intent)
        all_preds.append(pred.cpu().numpy())
        all_gt.append(target.cpu().numpy())

all_preds = np.vstack(all_preds)
all_gt    = np.vstack(all_gt)
errors    = np.linalg.norm(all_preds - all_gt, axis=1)

# compute final ADE/FDE
ade = errors.mean()
fde = errors.mean()
print(f"✅ ADE: {ade:.4f} | FDE: {fde:.4f}")

# ── 4) Visualization ───────────────────────────────────────────────────────
# 4.1 True vs Predicted X
plt.figure(figsize=(6,6))
plt.scatter(all_gt[:,0], all_preds[:,0], alpha=0.3)
mn, mx = all_gt[:,0].min(), all_gt[:,0].max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel('True X')
plt.ylabel('Predicted X')
plt.title('True vs Predicted X')
plt.tight_layout()

# 4.2 Error histogram
plt.figure(figsize=(6,4))
plt.hist(errors, bins=50, alpha=0.7)
plt.xlabel('Euclidean Error')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()

# 4.3 2D overlay of (x,y)
plt.figure(figsize=(6,6))
plt.scatter(all_gt[:,0], all_gt[:,1], c='blue', alpha=0.3, label='True')
plt.scatter(all_preds[:,0], all_preds[:,1], c='red',  alpha=0.3, label='Pred')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('True (blue) vs Predicted (red) Trajectories')
plt.tight_layout()

plt.show()

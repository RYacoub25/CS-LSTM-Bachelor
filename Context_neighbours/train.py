import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from dataset import load_and_preprocess
from model import ContextualSocialLSTM, apply_weight_init

# paths
ego_path        = 'data/processed/ego_vehicle_with_intention.csv'
social_path     = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'

# load & preprocess with radius masking
X_hist, X_nbrs, X_mask, y, ctx, intent = load_and_preprocess(
    ego_path, social_path, contextual_path,
    seq_len=30, target_step=30,
    grid_size=3.0, max_neighbors=10,
    radius=3.5
)

# add time‐step dim to nbrs: (N,10,9)→(N,10,1,9)
X_nbrs = np.expand_dims(X_nbrs, axis=2)

# align lengths
y = y.astype(np.float32)
min_len = min(
    len(X_hist), len(X_nbrs), len(X_mask),
    len(y),    len(ctx),    len(intent)
)
X_hist, X_nbrs, X_mask, y, ctx, intent = [
    arr[:min_len] for arr in
    (X_hist, X_nbrs, X_mask, y, ctx, intent)
]

# train/test split (six arrays)
splits = train_test_split(
    X_hist, X_nbrs, X_mask, ctx, y, intent,
    test_size=0.2, random_state=42
)
h_tr,   h_te,   nbr_tr, nbr_te,   m_tr,   m_te, \
ctx_tr, ctx_te, y_tr,   y_te,     it_tr,  it_te = splits

# DataLoaders
train_ds = TensorDataset(
    torch.tensor(h_tr,   dtype=torch.float32),
    torch.tensor(nbr_tr, dtype=torch.float32),
    torch.tensor(m_tr,   dtype=torch.float32),
    torch.tensor(ctx_tr, dtype=torch.float32),
    torch.tensor(y_tr,   dtype=torch.float32),
    torch.tensor(it_tr,  dtype=torch.float32),
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

test_ds = TensorDataset(
    torch.tensor(h_te,   dtype=torch.float32),
    torch.tensor(nbr_te, dtype=torch.float32),
    torch.tensor(m_te,   dtype=torch.float32),
    torch.tensor(ctx_te, dtype=torch.float32),
    torch.tensor(y_te,   dtype=torch.float32),
    torch.tensor(it_te,  dtype=torch.float32),
)
test_dl = DataLoader(test_ds, batch_size=64)

# model
model     = ContextualSocialLSTM(context_dim=ctx.shape[1], intention_dim=1)
apply_weight_init(model)
opt       = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# train
for epoch in range(10):
    model.train()
    tot = 0.0
    for hist, nbrs, mask, context, target, it in train_dl:
        it = it.unsqueeze(1)
        pred = model(hist, nbrs, mask, context, it)
        loss = criterion(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    print(f"Epoch {epoch+1} | Loss {tot/len(train_dl):.4f}")

# eval
model.eval()
ades, fdes = [], []
with torch.no_grad():
    for hist, nbrs, mask, context, target, it in test_dl:
        it = it.unsqueeze(1)
        pred = model(hist, nbrs, mask, context, it)
        err  = torch.norm(pred - target, dim=1)
        ades.append(err.mean().item())
        fdes.append(err.mean().item())

print(f"✅ ADE: {np.mean(ades):.4f} | FDE: {np.mean(fdes):.4f}")

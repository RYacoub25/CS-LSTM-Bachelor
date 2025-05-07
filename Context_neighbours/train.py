

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from dataset import load_and_preprocess
from model import ContextualSocialLSTM, apply_weight_init
from torch import nn

# paths
ego_path        = 'data/processed/ego_vehicle_with_intention.csv'
social_path     = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'

# load & preprocess (now returns 8 arrays)
(
    X_hist, X_nbrs, X_mask,
    y, ctx, intention,
    nbr_pos, ego_pos
) = load_and_preprocess(ego_path, social_path, contextual_path)

# add time-dim to neighbors: (N,10,9)→(N,10,1,9)
X_nbrs = np.expand_dims(X_nbrs, axis=2)

# train/test split (include nbr_pos & ego_pos)
splits = train_test_split(
    X_hist, X_nbrs, X_mask, ctx, y, intention, nbr_pos, ego_pos,
    test_size=0.2, random_state=42
)
(
    h_tr,   h_te,
    nbr_tr, nbr_te,
    m_tr,   m_te,
    ctx_tr, ctx_te,
    y_tr,   y_te,
    int_tr, int_te,
    np_tr,  np_te,
    ep_tr,  ep_te
) = splits

# DataLoaders
train_ds = TensorDataset(
    torch.tensor(h_tr,   dtype=torch.float32),  # hist
    torch.tensor(nbr_tr, dtype=torch.float32),  # nbrs
    torch.tensor(m_tr,   dtype=torch.float32),  # mask
    torch.tensor(ctx_tr, dtype=torch.float32),  # contextual
    torch.tensor(y_tr,   dtype=torch.float32),  # target
    torch.tensor(int_tr, dtype=torch.float32),  # intention
    torch.tensor(np_tr,  dtype=torch.float32),  # nbr_pos
    torch.tensor(ep_tr,  dtype=torch.float32),  # ego_pos
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

test_ds = TensorDataset(
    torch.tensor(h_te,   dtype=torch.float32),
    torch.tensor(nbr_te, dtype=torch.float32),
    torch.tensor(m_te,   dtype=torch.float32),
    torch.tensor(ctx_te, dtype=torch.float32),
    torch.tensor(y_te,   dtype=torch.float32),
    torch.tensor(int_te, dtype=torch.float32),
    torch.tensor(np_te,  dtype=torch.float32),
    torch.tensor(ep_te,  dtype=torch.float32),
)
test_dl = DataLoader(test_ds, batch_size=64)

# model
model     = ContextualSocialLSTM(
    context_dim=ctx.shape[1],
    intention_dim=1,
    dist_temp=1.0
)
apply_weight_init(model)
opt       = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# training
for epoch in range(10):
    model.train()
    total = 0.0
    for hist, nbrs, mask, context, target, intent, nbrp, egop in train_dl:
        intent = intent.unsqueeze(1)
        pred   = model(hist, nbrs, mask, context, intent, nbrp, egop)
        loss   = criterion(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total/len(train_dl):.4f}")

torch.save(model.state_dict(), "contextual_social_neighbor_lstm.pth")
print("✅ Model saved as 'contextual_social_neighbor_lstm.pth'")
# evaluation
model.eval()
ades, fdes = [], []
with torch.no_grad():
    for hist, nbrs, mask, context, target, intent, nbrp, egop in test_dl:
        intent = intent.unsqueeze(1)
        pred   = model(hist, nbrs, mask, context, intent, nbrp, egop)
        err    = torch.norm(pred - target, dim=1)
        ades.append(err.mean().item())
        fdes.append(err.mean().item())

print(f"✅ ADE: {np.mean(ades):.4f} | FDE: {np.mean(fdes):.4f}")

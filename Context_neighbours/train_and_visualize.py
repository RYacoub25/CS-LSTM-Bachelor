import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess
from model import ContextualSocialLSTM, apply_weight_init
from torch import nn

# Load and preprocess data
ego_path = 'data/processed/ego_vehicle_with_intention.csv'
social_path = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'

X_hist, X_nbrs, X_mask, y, ctx, intention, nbr_pos, ego_pos = load_and_preprocess(
    ego_path, social_path, contextual_path
)
# Add time dimension for neighbors
X_nbrs = np.expand_dims(X_nbrs, axis=2)

# Align lengths and split
y = y.astype(np.float32)
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(ctx), len(intention))
X_hist, X_nbrs, X_mask, y, ctx, intention = (
    X_hist[:min_len], X_nbrs[:min_len], X_mask[:min_len],
    y[:min_len], ctx[:min_len], intention[:min_len]
)
splits = train_test_split(
    X_hist, X_nbrs, X_mask, ctx, y, intention,
    nbr_pos, ego_pos,
    test_size=0.2, random_state=42
)
h_tr, h_te, nbr_tr, nbr_te, m_tr, m_te, ctx_tr, ctx_te, y_tr, y_te, int_tr, int_te, np_tr, np_te, ep_tr, ep_te = splits

# Dataloaders
train_ds = TensorDataset(
    torch.tensor(h_tr, dtype=torch.float32),
    torch.tensor(nbr_tr, dtype=torch.float32),
    torch.tensor(m_tr, dtype=torch.float32),
    torch.tensor(ctx_tr, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.float32),
    torch.tensor(int_tr, dtype=torch.float32).unsqueeze(1),
    torch.tensor(np_tr, dtype=torch.float32),
    torch.tensor(ep_tr, dtype=torch.float32),
)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

test_ds = TensorDataset(
    torch.tensor(h_te, dtype=torch.float32),
    torch.tensor(nbr_te, dtype=torch.float32),
    torch.tensor(m_te, dtype=torch.float32),
    torch.tensor(ctx_te, dtype=torch.float32),
    torch.tensor(y_te, dtype=torch.float32),
    torch.tensor(int_te, dtype=torch.float32).unsqueeze(1),
    torch.tensor(np_te, dtype=torch.float32),
    torch.tensor(ep_te, dtype=torch.float32),
)
test_dl = DataLoader(test_ds, batch_size=64)

# Initialize model
model = ContextualSocialLSTM(context_dim=ctx.shape[1], intention_dim=1, dist_temp=1.0)
apply_weight_init(model)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()

# Train (10 epochs)
for epoch in range(10):
    model.train()
    total_loss = 0
    for hist, nbrs, mask, context, target, intent, nbrp, egop in train_dl:
        pred = model(hist, nbrs, mask, context, intent, nbrp, egop)
        loss = criterion(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dl):.4f}")

# Evaluate and collect metrics
model.eval()
errors, nbr_counts, intentions, context_mags, speeds = [], [], [], [], []
traj_examples = []

with torch.no_grad():
    for i, (hist, nbrs, mask, context, target, intent, nbrp, egop) in enumerate(test_dl):
        pred = model(hist, nbrs, mask, context, intent, nbrp, egop)
        err = torch.norm(pred - target, dim=1).cpu().numpy()
        errors.extend(err)
        nbr_counts.extend(mask.sum(dim=(1,2)).cpu().numpy())
        intentions.extend(intent.squeeze(1).cpu().numpy())
        context_mags.extend(torch.norm(context, dim=1).cpu().numpy())
        # speed = norm of velocity at last history step
        hist_np = hist.cpu().numpy()
        speeds.extend(np.linalg.norm(hist_np[:,:,-6:-3], axis=2).mean(axis=1))
        # store first example for trajectory plot
        if i == 0:
            traj_examples = {
                'hist': hist[0].cpu().numpy(),
                'true': target[0].cpu().numpy(),
                'pred': pred[0].cpu().numpy(),
                'nbrs': nbrs[0,:,0,:2].cpu().numpy(),  # neighbor tracks at current time
                'egop': egop[0].cpu().numpy()
            }

# Create DataFrame for analysis
df = pd.DataFrame({
    'error': errors,
    'nbr_count': nbr_counts,
    'intention': intentions,
    'context_mag': context_mags,
    'speed': speeds
})

# 1. Plot example trajectory
plt.figure()
hist_xy = traj_examples['hist'][:,:2]
plt.plot(hist_xy[:,0], hist_xy[:,1], label='History')
plt.scatter([traj_examples['true'][0]], [traj_examples['true'][1]], label='True Future')
plt.scatter([traj_examples['pred'][0]], [traj_examples['pred'][1]], label='Pred Future')
for nbr in traj_examples['nbrs']:
    plt.scatter(nbr[0], nbr[1], marker='x')
plt.scatter(traj_examples['egop'][0], traj_examples['egop'][1], marker='o', label='Ego Pos')
plt.title('Example Trajectory')
plt.legend()
plt.show()

# 2. ADE by intention
ade_by_intent = df.groupby('intention')['error'].mean()
plt.figure()
plt.bar(ade_by_intent.index.astype(str), ade_by_intent.values)
plt.title('ADE by Intention Class')
plt.xlabel('Intention (0=Straight,1=Left,-1=Right)')
plt.ylabel('Mean ADE')
plt.show()

# 3. Scatter: neighbor count vs error
plt.figure()
plt.scatter(df['nbr_count'], df['error'], s=2)
plt.title('Error vs. Neighbor Count')
plt.xlabel('Number of Neighbors')
plt.ylabel('Error')
plt.show()

# 4. Histograms: speed, neighbor count, context magnitude
plt.figure()
plt.hist(df['speed'], bins=30)
plt.title('Histogram of Ego Speeds')
plt.xlabel('Speed')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(df['nbr_count'], bins=10)
plt.title('Histogram of Neighbor Counts')
plt.xlabel('Neighbor Count')
plt.ylabel('Count')
plt.show()

plt.figure()
plt.hist(df['context_mag'], bins=30)
plt.title('Histogram of Context Magnitudes')
plt.xlabel('Context Vector Norm')
plt.ylabel('Count')
plt.show()

# 5. Stub: neighbor-attention vs distance (requires model to return weights)
# (This will be implemented once the model is updated to return attention weights.)

# Summary table
print(df.describe())

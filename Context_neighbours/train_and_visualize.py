import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataset import load_and_preprocess
from model import ContextualSocialLSTM, apply_weight_init
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
ego_path = 'data/processed/ego_vehicle_with_intention.csv'
social_path = 'data/processed/social_vehicles.csv'
contextual_path = 'data/processed/contextual_features_merged.npy'
X_hist, X_nbrs, X_mask, y, ctx, intention, nbr_pos, ego_pos = load_and_preprocess(
     ego_path, social_path, contextual_path)
# align lengths & split
min_len = min(len(X_hist), len(X_nbrs), len(X_mask), len(y), len(ctx), len(intention))
X_hist, X_nbrs, X_mask, y, ctx, intention = (
    X_hist[:min_len], X_nbrs[:min_len], X_mask[:min_len],
    y[:min_len], ctx[:min_len], intention[:min_len]
)
X_hist_train, X_hist_test, y_train, y_test, ctx_train, ctx_test, Xn_train, Xn_test, Xm_train, Xm_test, intent_train, intent_test = train_test_split(
    X_hist, y, ctx, X_nbrs, X_mask, intention, test_size=0.2, random_state=42
)

# DataLoader
test_ds = TensorDataset(
    torch.tensor(X_hist_test, dtype=torch.float32),
    torch.tensor(Xn_test, dtype=torch.float32),
    torch.tensor(Xm_test, dtype=torch.float32),
    torch.tensor(ctx_test, dtype=torch.float32),
    torch.tensor(y_test[:, :2], dtype=torch.float32),
    torch.tensor(intent_test, dtype=torch.float32)
)
test_dl = DataLoader(test_ds, batch_size=64)

# 2. Load model
model = ContextualSocialLSTM(context_dim=ctx.shape[1], intention_dim=1)
apply_weight_init(model)
model.load_state_dict(torch.load("contextual_social_neighbor_lstm.pth"))
model.eval()

# 3. Pick one batch
hist, nbrs, mask, context, target, intent = next(iter(test_dl))
with torch.no_grad():
    pred = model(hist, nbrs, mask, context, intent.unsqueeze(1))

# 4. Plot sample trajectories
def plot_one_example(idx):
    h = hist[idx].numpy()
    t = target[idx].numpy()
    p = pred[idx].numpy()
    nbr = nbrs[idx].numpy()
    plt.figure()
    plt.plot(h[:,0], h[:,1], label='history')
    plt.plot(t[:,0], t[:,1], label='ground truth')
    plt.plot(p[:,0], p[:,1], linestyle='--', label='predicted')
    for n in nbr:
        plt.plot(n[:,0], n[:,1], alpha=0.3)
    plt.legend()
    plt.title(f"Example {idx}")
    plt.show()

plot_one_example(0)
plot_one_example(1)

# 5. Error histogram by intention and neighbor count
errors = []
int_list = []
nbr_counts = []
for h, n, m, c, tgt, it in test_dl:
    it = it.numpy()
    with torch.no_grad():
        p = model(h, n, m, c, torch.tensor(it).unsqueeze(1))
    e = torch.norm(p - torch.tensor(tgt), dim=1).numpy()
    errors.append(e)
    int_list.append(it)
    nbr_counts.append(m.sum(dim=1).numpy())
errors = np.concatenate(errors)
int_list = np.concatenate(int_list)
nbr_counts = np.concatenate(nbr_counts)

# Boxplot by intention
plt.figure()
groups = { -1:'Right Turn', 0:'Straight', 1:'Left Turn' }
data = [ errors[int_list==val] for val in [-1,0,1] ]
plt.boxplot(data, labels=[groups[v] for v in [-1,0,1]])
plt.title("ADE by Intention")
plt.ylabel("ADE")
plt.show()

# Scatter: ADE vs. neighbor count
plt.figure()
plt.scatter(nbr_counts, errors, alpha=0.3)
plt.xlabel("Number of Neighbors")
plt.ylabel("ADE")
plt.title("ADE vs Neighbor Count")
plt.show()

# 6. Error over horizon
# (Not available: model predicts only final point)

# 7. Data sanity checks
# 7a. Speed distribution
speeds = np.linalg.norm(X_hist_test[:,:,3:5], axis=2).flatten()
plt.figure()
plt.hist(speeds, bins=50)
plt.title("Distribution of Ego Speed Magnitudes (historical)")
plt.xlabel("Speed")
plt.ylabel("Count")
plt.show()

# 7b. Neighbor count distribution
all_counts = X_mask_test = Xm_test = Xm_test = None  # using nbr_counts from above
plt.figure()
plt.hist(nbr_counts, bins=range(12))
plt.title("Distribution of Neighbor Counts")
plt.xlabel("Count")
plt.ylabel("Frames")
plt.show()

# 7c. Context feature magnitude
ctx_mag = np.linalg.norm(ctx_test, axis=1)
plt.figure()
plt.hist(ctx_mag, bins=50)
plt.title("Distribution of Context Feature Magnitudes")
plt.xlabel("||ctx||")
plt.ylabel("Count")
plt.show()

# 8. (Optional) Neighbor-attention weights vs distance
# Requires model to output attention weights. If available:
# distances = np.linalg.norm(nbr_pos_test - ego_pos_test[:,None,:], axis=2).flatten()
# weights = neighbor_weights.flatten()
# plt.figure()
# plt.scatter(distances, weights, alpha=0.3)
# plt.xlabel("Distance")
# plt.ylabel("Attention Weight")
# plt.title("Neighbor Attention vs Distance")
# plt.show()

# That’s a first pass at the visualization suite.

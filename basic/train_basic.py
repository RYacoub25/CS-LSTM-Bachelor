import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from highwayNet_basic import highwayNet

# ==== Load Data ====
df = pd.read_csv('data/processed/ego_vehicle.csv')

# ==== Clean Data ====
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ==== Normalize Features ====
time_dependent_columns = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az']
scaler = StandardScaler()
df[time_dependent_columns] = scaler.fit_transform(df[time_dependent_columns])

# ==== Create Sequences (for 30-step prediction) ====
def create_sequences(df, seq_len=30, target_steps=30):
    sequences = []
    targets = []
    for i in range(len(df) - seq_len - target_steps):
        seq = df[time_dependent_columns].iloc[i:i+seq_len].values
        target = df[['x', 'y']].iloc[i + seq_len + target_steps - 1].values
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)



X, y = create_sequences(df, seq_len=30, target_steps=30)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Load Model ====
args = {
    'out_length': 30,
    'encoder_size': 128,
    'decoder_size': 128,
    'use_cuda': torch.cuda.is_available()
}

model = highwayNet(args)
model.load_state_dict(torch.load('basic_lstm_no_social_no_context.pth'))
model.eval()

if args['use_cuda']:
    model.cuda()

# ==== Prepare Test Loader ====
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ==== Evaluation ====
ade_list = []
fde_list = []

with torch.no_grad():
    for inputs, targets in test_loader:
        if args['use_cuda']:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs, _ = model(inputs)  # shape: (batch_size, 30, 2)

        # targets should also be (batch_size, 30, 2)
        ade = torch.mean(torch.norm(outputs - targets, dim=1))  # ✅ Correct for [batch_size, 2]
        fde = torch.mean(torch.norm(outputs - targets, dim=1))

        ade_list.append(ade.item())
        fde_list.append(fde.item())

# ==== Print Results ====
print(f"\n\U0001F4CA Evaluation Results:")
print(f"\u2705 Average Displacement Error (ADE): {np.mean(ade_list):.4f}")
print(f"\u2705 Final Displacement Error (FDE): {np.mean(fde_list):.4f}")

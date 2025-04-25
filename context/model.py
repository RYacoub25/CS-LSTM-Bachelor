# model.py
import torch
import torch.nn as nn

class ContextualSocialLSTM(nn.Module):
    def __init__(self, input_size=18, context_dim=7, hidden_size=256, output_size=2, num_layers=2):
        super(ContextualSocialLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.context_fc = nn.Sequential(
            nn.Linear(context_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        self.output_fc = nn.Linear(hidden_size * 2, output_size)  # Combine LSTM + context

    def forward(self, hist, nbrs, mask, context):
        # Ensure hist and nbrs have the same sequence length
        if hist.size(1) != nbrs.size(1):
            max_len = max(hist.size(1), nbrs.size(1))
            if hist.size(1) < max_len:
                padding = torch.zeros(hist.size(0), max_len - hist.size(1), hist.size(2), device=hist.device)
                hist = torch.cat([hist, padding], dim=1)
            if nbrs.size(1) < max_len:
                padding = torch.zeros(nbrs.size(0), max_len - nbrs.size(1), nbrs.size(2), device=nbrs.device)
                nbrs = torch.cat([nbrs, padding], dim=1)

        x = torch.cat([hist, nbrs], dim=2)  # [B, T, 18]
        x, _ = self.lstm(x)
        lstm_out = x[:, -1, :]  # [B, hidden]

        context_out = self.context_fc(context)  # [B, hidden]
        combined = torch.cat([lstm_out, context_out], dim=1)  # [B, hidden*2]

        output = self.output_fc(combined)  # [B, 2] → x, y
        return output

# Weight initialization

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def apply_weight_init(model):
    model.apply(init_weights)